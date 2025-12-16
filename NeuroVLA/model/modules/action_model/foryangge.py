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

