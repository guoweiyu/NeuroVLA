import math

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

# --- added: reset helper for snntorch states ---
def _reset_snn_states(module: nn.Module):
    # 尽量用库自带的 reset_state，如果没有就手动 detach/清空
    for m in module.modules():
        if isinstance(m, snn.Leaky):
            if hasattr(m, "reset_state") and callable(getattr(m, "reset_state")):
                m.reset_state()
            else:
                for attr in ("mem", "spk", "syn", "state"):
                    if hasattr(m, attr):
                        v = getattr(m, attr)
                        if torch.is_tensor(v):
                            setattr(m, attr, v.detach())
                        else:
                            setattr(m, attr, None)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        beta_hidden = torch.rand(dim)
        thr_hidden = torch.rand(dim)
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient

        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            snn.Leaky(beta=beta_hidden, init_hidden=True, spike_grad=spike_grad),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        spike = self.ffn(x)
        return spike

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()

        #snn参数初始化
        beta_in = torch.rand(hidden_dim)
        thr_in = torch.rand(hidden_dim)
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient
        beta_out = torch.rand(1)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad,init_hidden=True)
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    # --- added: expose a reset entrypoint on the model itself ---
    def reset_state(self):
        _reset_snn_states(self)

    def forward(self, x):
        # 在每次前向传递时重置 SNN 状态，避免跨图保留导致的二次反传错误
        self.reset_state()

        # x: (batch_size, input_dim)
        mem_3 = self.li_out.init_leaky()
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.lif_in(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        _, mem_3 = self.li_out(x, mem_3)
        output=self.fc3(mem_3)
        return output

class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, actions_hidden_states.shape[1], -1)
        
        action = self.model(rearranged_actions_hidden_states)
        return action
    
def get_action_model(input_dim=768,hidden_dim=768, action_dim=7):
    """
    根据配置创建 ActionModel 实例
    :param config: 包含模型参数的配置字典或对象
    :return: ActionModel 实例
    """
    action_head = L1RegressionActionHead(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    
    return action_head

class FiLMedActionStateModulator(nn.Module):
    """
    Applies FiLM-style modulation to action hidden states conditioned on historical robot states.
    """
    def __init__(
        self,
        action_hidden_dim: int,
        robot_state_dim: int,
        projector_hidden_dim: int = 512,
    ):
        super().__init__()
        self.gamma_projector = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )
        self.beta_projector = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )

    def forward(self, actions_hidden_states: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: (batch_size, seq_len, action_hidden_dim)
            robot_states: (batch_size, history_len, robot_state_dim)

        Returns:
            torch.Tensor: Modulated action hidden states, same shape as input.
        """
        batch_size, _, action_hidden_dim = actions_hidden_states.shape
        # Aggregate historical robot states to a single conditioning vector.
        # Here we use mean pooling; replace with a learned encoder if needed.
        pooled_robot_state = robot_states.mean(dim=1)  # (batch_size, robot_state_dim)

        gamma = self.gamma_projector(pooled_robot_state)  # (batch_size, action_hidden_dim)
        beta = self.beta_projector(pooled_robot_state)    # (batch_size, action_hidden_dim)

        gamma = gamma.view(batch_size, 1, action_hidden_dim)
        beta = beta.view(batch_size, 1, action_hidden_dim)

        modulated_actions = actions_hidden_states * (1 + gamma) + beta
        return modulated_actions

class GRU_GatedFiLModulator(nn.Module):
    """
    【V5版 - GRU 编码器 + 门控融合】
    1. 使用 GRU 智能编码 T=16 的机器人历史状态，捕获时序信息。
    2. 使用 V3 的门控 (Gating) 架构，将 GRU 总结的状态向量
       与高维 VLM 意图向量进行安全、高效的融合。
    """
    def __init__(
        self,
        action_hidden_dim: int,     # VLM 特征 (e.g., 768)
        robot_state_dim: int,         # 机器人状态 (e.g., 8)
        projector_hidden_dim: int = 512,
        gru_hidden_dim: int = 64,     # GRU 的隐藏维度
    ):
        super().__init__()
        
        # --- 1. 新增: GRU 状态编码器 ---
        # 它将 (B, T_hist, D_robot) -> (B, D_gru_hidden)
        self.robot_state_encoder = nn.GRU(
            input_size=robot_state_dim,   # 8
            hidden_size=gru_hidden_dim,   # 64
            num_layers=2,                 # 2层 GRU 足够
            batch_first=True,             # 接受 (B, T, D)
            bidirectional=False,          # 历史信息是单向的
        )
        
        # --- 2. 预投影器 (来自 V3) ---
        
        # 投影器A: 处理 VLM 意图
        # (B, D_action) -> (B, D_hidden)
        self.action_pre_projector = nn.Sequential(
            nn.LayerNorm(action_hidden_dim),
            nn.Linear(action_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )
        
        # 投影器B: 处理 GRU 编码后的状态
        # (B, D_gru_hidden) -> (B, D_hidden)
        self.robot_state_pre_projector = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim), # 注意: LN 的维度是 GRU 的输出
            nn.Linear(gru_hidden_dim, projector_hidden_dim),
            nn.ReLU(),
        )
        
        # --- 3. 门控投影器 (来自 V3) ---
        # (B, D_hidden) -> (B, D_hidden)
        self.gate_projector = nn.Sequential(
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.Sigmoid()
        )

        # --- 4. 最终的 gamma/beta 投影器 (来自 V3) ---
        fused_input_dim = projector_hidden_dim + projector_hidden_dim
        
        self.gamma_projector = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )
        
        self.beta_projector = nn.Sequential(
            nn.LayerNorm(fused_input_dim),
            nn.Linear(fused_input_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, action_hidden_dim),
        )

    def forward(self, actions_hidden_states: torch.Tensor, robot_states: torch.Tensor,save_ga_be=False,savepath="/workspace/llavavla0/results/Visulazition/gamma_beta/gamma_beta.pt") -> torch.Tensor:
        batch_size, _, action_hidden_dim = actions_hidden_states.shape

        # --- 步骤 1: 智能编码历史状态 ---
        # robot_states shape: (B, T_hist=16, D_robot=8)
        # _, h_n = self.robot_state_encoder(robot_states)
        # h_n shape: (num_layers, B, D_gru_hidden)
        #
        # 我们只取最后一层的隐藏状态
        # pooled_robot_state = h_n[-1] # shape: (B, D_gru_hidden)
        
        # [更正] 推荐使用 output 的最后一个时间步，h_n 有时在 DDP 下会出问题
        # output shape: (B, T_hist, D_gru_hidden)
      # --- 步骤 1: 智能编码历史状态 (强制 FP32 + contiguous) ---
        # --- Step 1: 智能编码历史状态 (匹配 GRU dtype + contiguous) ---
        original_dtype = actions_hidden_states.dtype

        # 确保 device 一致
        robot_states_device = actions_hidden_states.device

        # 获取 GRU 权重的 dtype（可能是 bf16/fp16/fp32）
        gru_dtype = next(self.robot_state_encoder.parameters()).dtype

        # 转换 robot_states 到 GRU 权重的 dtype
        robot_states_cast = robot_states.to(dtype=gru_dtype, device=robot_states_device)
        if not robot_states_cast.is_contiguous():
            robot_states_cast = robot_states_cast.contiguous()

        # 在禁用 autocast 的上下文中执行（防止 cudnn 类型不匹配）
        with torch.autocast("cuda", enabled=False):
            output, _ = self.robot_state_encoder(robot_states_cast)

        # 转回模型原始精度（例如 fp16）
        output = output.to(original_dtype)
        pooled_robot_state = output[:, -1, :]  # (B, D_gru_hidden)
        

        # --- 步骤 2: 聚合 VLM 意图 ---
        pooled_action_state = actions_hidden_states.mean(dim=1) # (B, D_action)

        # --- 步骤 3: 投影 + 门控 (来自 V3) ---
        action_proj = self.action_pre_projector(pooled_action_state)     # (B, D_hidden)
        robot_state_proj = self.robot_state_pre_projector(pooled_robot_state) # (B, D_hidden)
        
        gate = self.gate_projector(robot_state_proj)
        gated_action_proj = action_proj * gate

        fused_vector = torch.cat([gated_action_proj, robot_state_proj], dim=-1)

        # --- 步骤 4: 生成 FiLM 参数 ---
        gamma = self.gamma_projector(fused_vector)
        beta = self.beta_projector(fused_vector)

        gamma = gamma.view(batch_size, 1, action_hidden_dim)
        beta = beta.view(batch_size, 1, action_hidden_dim)
        if save_ga_be:
            torch.save({
                "gamma": gamma.detach().cpu(),
                "beta": beta.detach().cpu()
            }, savepath)
        modulated_actions = actions_hidden_states * (1 + gamma) + beta
        return modulated_actions

        modulated_actions = actions_hidden_states * (1 + gamma) + beta
        return modulated_actions

def get_gruedit_model(input_dim=768,hidden_dim=768, robot_state_dim=8):
    edit_head = GRU_GatedFiLModulator(action_hidden_dim=input_dim, robot_state_dim=robot_state_dim,  projector_hidden_dim=hidden_dim)

    return edit_head



def get_edit_model(input_dim=768,hidden_dim=768, robot_state_dim=8):
    edit_head = FiLMedActionStateModulator(action_hidden_dim=input_dim, robot_state_dim=robot_state_dim,  projector_hidden_dim=hidden_dim)

    return edit_head

if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 设置测试参数
    batch_size = 2  # B
    sequence_length = 16  # 序列长度
    hidden_dim = 768  # 隐藏状态维度
    device = torch.device("cuda:0")
    # 创建测试输入 [B, 16, 768]
    test_hidden_states = torch.randn(batch_size, sequence_length, hidden_dim).to(device)
    print(f"输入隐藏状态形状: {test_hidden_states.shape}")
    
    # 创建action model
    

    action_model = get_action_model(input_dim=768, hidden_dim=768*2, action_dim=7).to(device)
    print(f"Action model: {action_model}")
    
    # 测试预测
    with torch.no_grad():
        predicted_actions = action_model.predict_action(test_hidden_states)
        print(f"预测动作形状: {predicted_actions.shape}")
        print(f"预测动作: \n{predicted_actions}")
    
    print("测试完成！")

