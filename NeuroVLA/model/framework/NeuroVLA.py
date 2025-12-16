from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


from NeuroVLA.model.modules.action_model.mlp_action_model import get_action_model, get_edit_model, get_gruedit_model

from NeuroVLA.model.framework.base_framework import baseframework
from NeuroVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from NeuroVLA.model.modules.projector.QFormer import get_layerwise_qformer
from NeuroVLA.model.modules.vlm import get_vlm_model
from NeuroVLA.model.tools import FRAMEWORK_REGISTRY
from NeuroVLA.training.trainer_utils.trainer_tools import resize_images

# from NeuroVLA.model.modules.action_model.spike_action_model import get_action_model, get_edit_model


@FRAMEWORK_REGISTRY.register("spikevla_xiaonao_yibu")
class SpikeVLA_MLP(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.layer_qformer = get_layerwise_qformer(config=self.config)
        self.action_model = get_action_model(input_dim=768, hidden_dim=768 * 2, action_dim=7)
        # self.edit_model = get_edit_model(input_dim=768, hidden_dim=256, robot_state_dim=8)
        self.edit_model = get_gruedit_model(input_dim=768, hidden_dim=256, robot_state_dim=8)
        self.L1_loss = nn.L1Loss()

        self.norm_stats = norm_stats

    def forward(
        self,
        examples: List[dict] = None,
        repeated_diffusion_steps: int = 4,
        **kwargs,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        inference_num = 0
        images = [example["image"] for example in examples]  #  [B， PLT]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        states = [example["state"] for example in examples]  # label [B， len, 8]
        if "solution" in examples[0]:
            solutions = [example["solution"] for example in examples]  # [B, dict]
        else:
            solutions = None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, solutions=solutions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            pass
            # dist.barrier()
        vlm_cot_loss = qwenvl_outputs.loss

        if vlm_cot_loss is None or torch.isnan(vlm_cot_loss):
            vlm_cot_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)
        # action_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1
            action_latent_feature = self.layer_qformer(
                qwenvl_outputs.hidden_states[start_layer:end_layer]
            )  # [B, 64, D_action]

            # here in real world robotics states should update in real time.
            states = torch.tensor(
                np.array(states), dtype=torch.float32, device=action_latent_feature.device
            )  # [B, 16, 8]
            all_predicted_actions = []
            inference_num = 0

            # inferencenum can be defined from 2 to 10.
            while inference_num < 2:
                edit_action_feature = self.edit_model(action_latent_feature, states)  # [B, 8, D_action]

                predicted_actions = self.action_model.predict_action(edit_action_feature)  # [B, 4, 7]
                all_predicted_actions.append(predicted_actions)

                predicted_states = torch.zeros_like(states)
                predicted_states[:, : predicted_actions.shape[1], :7] = predicted_actions
                predicted_states[:, :, 7] = states[:, :, 7]

                states = predicted_states.clone()

                inference_num += 1

            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=predicted_actions.device)
            predicted_action_tensor = torch.cat(all_predicted_actions, dim=1)
            action_loss = self.L1_loss(predicted_action_tensor, action_tensor)

        return {"action_loss": action_loss}

    def predict_action(  #
        self,
        batch_images: Union[Image, List[Image]],
        instructions: List[str],
        states: Optional[List[Sequence[float]]] = None,
        solutions: Union[Dict, List[Dict]] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        gabesave=False,
        savepath="/workspace/llavavla0/results/Visulazition/gamma_beta/gamma_beta.pt",
        **kwargs: str,
    ) -> np.ndarray:
        """
        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        predict_num = 0

        batch_images = resize_images(batch_images, target_size=(224, 224))  # list of PIL RGB for one instruction

        inferface_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

        all_predicted_actions = []
        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fmt: off
            qwenvl_outputs = self.qwen_vl_interface( 
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, 
                image_grid_thw =qwen_inputs.image_grid_thw,
                labels= qwen_inputs.input_ids.clone(),
                output_hidden_states=True, 
                return_dict=True,
            )

        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1

            action_latent_feature = self.layer_qformer(
                qwenvl_outputs.hidden_states[start_layer:end_layer]
            )  # [B, 64, D_action]

            using_cfg = cfg_scale > 1.0

            # model_dtype = next(self.action_model.net.parameters()).dtype
            B = action_latent_feature.shape[0]

            # states = torch.tensor(np.array(states), dtype=torch.float32, device=action_latent_feature.device)
            states = torch.tensor(
                np.array(states, dtype=np.float32),  # Ensure states is a numeric array
                dtype=torch.float32,
                device=action_latent_feature.device,
            )
            while predict_num < 2:
                edit_action_feature = self.edit_model(
                    action_latent_feature, states, save_ga_be=gabesave, savepath=savepath
                )
                samples = self.action_model.predict_action(edit_action_feature)
                all_predicted_actions.append(samples)
                predicted_states = torch.zeros_like(states)
                predicted_states[:, : samples.shape[1], :7] = samples  # 填入动作对应的部分
                predicted_states[:, :, 7] = states[:, :, 7]  # 保留gripper或pad状态

                # 更新状态
                states = predicted_states.clone()
                predict_num += 1
        predicted_action_tensor = torch.cat(all_predicted_actions, dim=1)
        normalized_actions = predicted_action_tensor.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


def build_model_framework(config: dict = {}) -> SpikeVLA_MLP:
    model = SpikeVLA_MLP(config=config)

    return model


if __name__ == "__main__":
    import pickle

    from omegaconf import OmegaConf

    with open("/workspace/samples_states.pkl", "rb") as f:
        samples = pickle.load(f)
    device = torch.device("cuda:0")

    config_yaml = "path/to/your/checkpoint/config.yaml"

    cfg = OmegaConf.load(config_yaml)
    # cfg.framework.layer_qformer.num_query_tokens=8
    spikeVLA_MLP = SpikeVLA_MLP(cfg)
    spikeVLA_MLP = spikeVLA_MLP.to(device)
    base_action_loss = spikeVLA_MLP(examples=samples, use_base_loss=True, refinement_weight=1.0)

    # print(f"Base Action Loss: {base_action_loss}")
    images = [sample["image"] for sample in samples]
    instructions = [sample["lang"] for sample in samples]
    states = [sample["state"] for sample in samples]

    with torch.inference_mode():
        normalized_actions = spikeVLA_MLP.predict_action(
            batch_images=images,
            instructions=instructions,
            states=states,
            apply_refinement=True,
        )
    print(f"Refined Actions Shape: {normalized_actions}")
