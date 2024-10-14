import torch
import numpy as np
import gc
from functools import partial
from torch import nn
from cdam.util import class2idx
from cdam.Transformer_Explainability.baselines.ViT.ViT_orig_LRP import vit_large_patch8_224 as vit_orig_LRP
from cdam.Transformer_Explainability.baselines.ViT.ViT_explanation_generator import LRP
from cdam.Transformer_Explainability.baselines.ViT.ViT_new import VisionTransformer as ViT

def integrated_gradients(
    model,
    input,
    baseline=None,
    steps=50,
    target_idx=0,
):
    """Calculate integrated gradients for a given input and baseline"""
    if baseline is None:
        baseline = torch.zeros_like(input).cuda()
    # Scale input and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]
    # Get the gradients
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_()
        model.zero_grad()
        output = model(scaled_input)[:,target_idx]
        grad = torch.autograd.grad(output, scaled_input)[0]
        grads.append(grad)
    grads = torch.stack(grads)
    avg_grads = torch.mean(grads[1:], dim=0)
    integrated_grad = (input - baseline) * avg_grads
    return torch.mean(integrated_grad.squeeze(), axis=0).detach().cpu()

def smoothgrad(
    model,
    input,
    noise_level=0.15,
    nsamples=50,
    target_idx=0
):
    """Calculate smoothgrad for a given input"""
    stdev = noise_level * (input.max() - input.min())
    total_gradients = torch.zeros_like(input)
    for _ in range(nsamples):
        noise = torch.randn_like(input) * stdev
        noisy_input = input + noise
        noisy_input.requires_grad_()
        model.zero_grad()
        output = model(noisy_input)[:, target_idx]
        grad = torch.autograd.grad(output, noisy_input)[0]
        total_gradients += grad
    total_gradients = torch.mean(total_gradients.squeeze(), axis=0).detach().cpu()
    return total_gradients.squeeze() / nsamples

def input_times_gradient(
    model,
    input,
    target_idx=0
):
    """Calculate input times gradient for a given input"""
    input.requires_grad_()
    model.zero_grad()
    output = model(input)[:, target_idx]
    grad = torch.autograd.grad(output, input)[0]
    return torch.mean((input * grad).squeeze(), axis=0).detach().cpu()

def token_ablation_map(model, img, target_idx, patch_size=8):
    """Calculate the change in prediction when each patch embedding is removed."""
    pred_delta = []
    pred_orig = model(img)
    B, nc, w, h = img.shape
    for i in range(int((w / patch_size) ** 2)):
        with torch.no_grad():
            # [i+1] to skip the CLS token
            pred = model(img, remove=[i + 1])
        pred_delta.append(float(pred_orig[:, target_idx] - pred[:, target_idx]))
    w = int(np.sqrt(len(pred_delta)))
    pred_delta = torch.tensor(pred_delta).reshape(w, w)
    pred_delta = torch.nn.functional.interpolate(
        pred_delta.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )
    return pred_delta.squeeze().detach().cpu()

# Relevance propagation (aka gradient times attention rollout)
# https://arxiv.org/abs/2103.15679 "Transformer Attribution/Relevance propagation/Grad×AttnRoll"
# Implementation from https://github.com/hila-chefer/Transformer-MM-Explainability

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index is None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32).cuda()
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward()
    one_hot.detach_()
    output.detach_()
    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        cam.detach_()
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    gc.collect()
    torch.cuda.empty_cache()
    return R[0, 1:]

# from baselines.ViT.ViT_new import vit_base_patch16_224 as vit


def load_rel_prop(model, num_classes):
    model_rel_prop = ViT(
        num_classes=num_classes,
        img_size=(224, 224),
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model_rel_prop.load_state_dict(model.state_dict(), strict=True)
    model_rel_prop.cuda()
    model_rel_prop.eval()
    return model_rel_prop


def generate_visualization(
    original_image, model_rel_prop, patch_size=8, class_index=None, return_raw=False
):
    transformer_attribution = generate_relevance(
        model_rel_prop, original_image, index=class_index
    ).detach()
    if return_raw:
        return transformer_attribution
    w = int(np.sqrt(transformer_attribution.shape))

    transformer_attribution = transformer_attribution.reshape(1, 1, w, w)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=patch_size, mode="nearest"
    )
    transformer_attribution = (
        transformer_attribution.reshape(w * patch_size, w * patch_size)
        .cuda()
        .data.cpu()
    )
    return transformer_attribution

def get_rel_prop_map(model, image, target_class=None, target_idx=None, return_raw=False):
    if target_idx != None:
        class_idx = target_idx
    elif target_class != None:
        class_idx = class2idx[target_class]
    # output = model(image.cuda())
    # print_top_classes(output)
    return generate_visualization(
        image,
        model_rel_prop=model,
        class_index=class_idx,
        return_raw=return_raw,
    )

#### Partial-LRP from https://arxiv.org/abs/1905.09418
#### Implementation from https://github.com/hila-chefer/Transformer-Explainability
def load_partial_lrp(model):
    model_orig_LRP = vit_orig_LRP(pretrained=True, state_dict=model.state_dict()).cuda()
    model_orig_LRP.eval()
    orig_LRP = LRP(model_orig_LRP)
    return orig_LRP, model_orig_LRP

def get_partial_lrp(img, lrp, target_idx, img_size=(224, 224), patch_size=8, return_raw=False):
    num_patches_x = int((img_size[0]/patch_size))
    partial_lrp = lrp.generate_LRP(img, method="last_layer", is_ablation=False, index=target_idx) 
    partial_lrp = partial_lrp.reshape(num_patches_x, num_patches_x)
    if return_raw:
        return partial_lrp.detach().cpu()
    partial_lrp = torch.nn.functional.interpolate(
        partial_lrp.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest"
    ).squeeze()
    return partial_lrp.detach().cpu()
