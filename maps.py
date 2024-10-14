# Definitions for the class-sensitive attention maps
import torch
import numpy as np
import copy
from torch import nn
from cdam.util import class2idx

def get_attention_map(model, sample_img, patch_size=8, head=None, return_raw=False):
    """This returns the attentions when CLS token is used as query in the last attention layer, averaged over all attention heads"""
    attentions = model.get_last_selfattention(sample_img)

    w_featmap = sample_img.shape[-2] // patch_size
    h_featmap = sample_img.shape[-1] // patch_size

    nh = attentions.shape[1]  # number of heads

    # this extracts the attention when cls is used as query
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    if return_raw:
        return torch.mean(attentions, dim=0).squeeze().detach().cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0]
    if head == None:
        mean_attention = torch.mean(attentions, dim=0).squeeze().detach().cpu()
        return mean_attention
    else:
        return attentions[head].squeeze().detach().cpu()
    
def get_concept_map(class_score, activation, grad, patch_size=8, return_raw=False):
    """Definition of class-sensitive attention maps (CDAM). The class_score can either be the activation of a neuron in the prediction vector or a similarity score between the latent representations of a concept and a sample"""
    class_score.backward()
    # Token 0 is CLS and others are 60x60 image patch tokens
    # print(grad)
    tokens = activation["last_att_in"][1:]
    grads = grad["last_att_in"][0][0, 1:]

    attention_scores = torch.tensor(
        [torch.dot(tokens[i], grads[i]) for i in range(len(tokens))]
    )
    if return_raw:
        return attention_scores
    else:
        w = int(np.sqrt(attention_scores.squeeze().shape[0]))
        attention_scores = attention_scores.reshape(w, w)
        return torch.nn.functional.interpolate(
            attention_scores.unsqueeze(0).unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        ).squeeze()
    
def get_concept_map_intgrad(class_score, activation, grad, baseline, patch_size=8, return_raw=False):
    """The only change is that we subtract the baseline from the tokens before computing the dot product with the gradients"""
    class_score.backward()
    # Token 0 is CLS and others are 60x60 image patch tokens
    tokens = activation["last_att_in"][1:]
    grads = grad["last_att_in"][0][0, 1:]

    # only consider the non-CLS tokens
    baseline = baseline[0, 1:]

    # baseline = torch.zeros_like(tokens[0]).cuda()

    attention_scores = torch.tensor(
        [torch.dot(tokens[i] - baseline[i], grads[i]) for i in range(len(tokens))]
    )

    if return_raw:
        return attention_scores
    else:
        w = int(np.sqrt(attention_scores.squeeze().shape[0]))
        attention_scores = attention_scores.reshape(w, w)

        return torch.nn.functional.interpolate(
            attention_scores.unsqueeze(0).unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        ).squeeze()
    
# To improve CDAM we obtain gradients using Integrated Gradients or SmoothGrad. For this we have to create a submodel that consists only of the last attention layer and the following classifier.

class NewModel(nn.Module):
    """This model is identical to the original model, but it starts at the final attention layer"""
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.block = copy.deepcopy(model.blocks[-1])
        self.norm = copy.deepcopy(model.norm)
        self.head = copy.deepcopy(model.head)

    def forward(self, x):
        x = self.block(x)
        x = self.norm(x)
        x = self.head(x)
        return x[:, 0]
    
def load_new_model(model):
    new_model = NewModel(model).cuda().eval()
    return new_model

# Define hooks necessary to obtain intermediate activations and gradients
# function to extract activation
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output[0].detach()

    return hook


# function to extract gradients
grad = {}


def get_gradient(name):
    def hook(model, input, output):
        grad[name] = output

    return hook


activation_block_in = {}


def get_activation_block_in(name):
    def hook(model, input, output):
        activation_block_in[name] = input[0].detach()

    return hook


new_activation = {}


def get_new_activation(name):
    def hook(model, input, output):
        new_activation[name] = output[0].detach()

    return hook


new_grad = {}


def get_new_gradient(name):
    def hook(model, input, output):
        new_grad[name] = output

    return hook

def create_hooks(model, new_model):
    """Set up hooks for gradient and activation extraction"""
    # We are calculating the gradients wrt the normalized inputs to the final attention layer
    # In the Dino implementation the normalization happens in the final block,
    # whereas in the original Transformer paper https://arxiv.org/pdf/1706.03762.pdf the normalization is done at the end of each transformer block
    try:
        activation_hook.remove()
        grad_hook.remove()
        activation_hook_block.remove()
        new_activation_hook.remove()
        new_grad_hook.remove()
    except:
        pass
    activation_hook = model.blocks[-1].norm1.register_forward_hook(
        get_activation("last_att_in")
    )

    grad_hook = model.blocks[-1].norm1.register_full_backward_hook(
        get_gradient("last_att_in")
    )

    activation_hook_block = model.blocks[-1].register_forward_hook(
        get_activation_block_in("last_att_in")
    )

    new_activation_hook = new_model.block.norm1.register_forward_hook(
        get_new_activation("last_att_in")
    )

    new_grad_hook = new_model.block.norm1.register_full_backward_hook(
        get_new_gradient("last_att_in")
    )

def concept_map_base(model, img, target_class=None, target_idx=None, return_raw=False):
    """Wrapper function that handles the forward pass to the model and target class/idx."""
    pred = model.forward(img)
    if target_idx != None:
        class_idx = target_idx
    elif target_class != None:
        class_idx = class2idx[target_class]
    class_attention_map = get_concept_map(
        class_score=pred[0][class_idx],
        activation=activation,
        grad=grad,
        return_raw=return_raw,
    )
    return class_attention_map

def concept_map_intgrad(model, trunc_model, img, target_class=None, target_idx=None, noise_level=0.0, return_raw=False):
    """The gradients for CDAM are calculated using integrated gradients technique."""
    if target_idx != None:
        class_idx = target_idx
    elif target_class != None:
        class_idx = class2idx[target_class]

    m = 50

    ig_maps = []
    model_out = model.forward(img)
    std = noise_level * (torch.max(activation_block_in["last_att_in"]) - torch.min(activation_block_in["last_att_in"]))
    for k in range(m):
        baseline = torch.zeros_like(activation_block_in["last_att_in"]).requires_grad_()
        # create a baseline with uniform noise instead of zeros
        # baseline = torch.rand_like(activation_block_in["last_att_in"]).requires_grad_()
        # baseline = 0.1 * baseline * torch.max(torch.abs(activation_block_in["last_att_in"]))
        
        noise = torch.normal(mean=0, std=std, size=activation_block_in["last_att_in"].shape).cuda()
        new_in = baseline + k / m * (activation_block_in["last_att_in"] - baseline) + noise
        new_model_out = trunc_model.forward(new_in)
        # new_model_out = trunc_model.forward( baseline )

        class_attention_map = get_concept_map_intgrad(
            class_score=new_model_out[0][class_idx],
            activation=new_activation,
            grad=new_grad,
            return_raw=return_raw,
            baseline=baseline,
        )
        ig_maps.append(class_attention_map)
        # plot_results(original=img[0][0].detach().cpu().numpy(), maps=[class_attention_map.detach().cpu()], figsize=(5,5))
    final_map = torch.mean(torch.stack(ig_maps[1:]), axis=0)
    return final_map

def concept_map_smoothgrad(model, trunc_model, img, target_class=None, target_idx=None, noise_level=0.05, return_raw=False):
    """The gradients for CDAM are calculated using SmoothGrad technique."""
    if target_idx != None:
        class_idx = target_idx
    elif target_class != None:
        class_idx = class2idx[target_class]

    m = 50

    # img_print = img.detach().cpu().numpy()[0]
    # img_print = np.moveaxis(img_print, 0, -1)

    # plt.imshow(img_print)

    smooth_maps = []
    model_out = model.forward(img)
    for k in range(m):
        # print(k)
        std = noise_level * (torch.max(activation_block_in["last_att_in"]) - torch.min(activation_block_in["last_att_in"]))
        noise = torch.normal(mean=0, std=std, size=activation_block_in["last_att_in"].shape).cuda()
        # create a baseline with uniform noise instead of zeros
        new_in = noise.requires_grad_() + activation_block_in["last_att_in"]
        new_model_out = trunc_model.forward(new_in)
        # new_model_out = trunc_model.forward( baseline )

        class_attention_map = get_concept_map(
            class_score=new_model_out[0][class_idx],
            activation=new_activation,
            grad=new_grad,
            return_raw=return_raw,
        )
        smooth_maps.append(class_attention_map)
        # plot_results(original=img[0][0].detach().cpu().numpy(), maps=[class_attention_map.detach().cpu()], figsize=(5,5))
    final_map = torch.mean(torch.stack(smooth_maps), axis=0)
    return final_map

def sensitivity_1_map(img, model, target_class, patch_size=8):
    """Calculate importance scores by removing each patch individually and measuring the change in the prediction score."""
    pred_delta = []
    pred_orig = model(img)
    B, nc, w, h = img.shape
    for i in range(int((w / patch_size) ** 2)):
        with torch.no_grad():
            # [i+1] to skip the CLS token
            pred = model(img, remove=[i + 1])
        pred_delta.append(float(pred_orig[:, target_class] - pred[:, target_class]))
    w = int(np.sqrt(len(pred_delta)))
    pred_delta = torch.tensor(pred_delta).reshape(w, w)
    pred_delta = torch.nn.functional.interpolate(
        pred_delta.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )
    return pred_delta.squeeze().detach().cpu()