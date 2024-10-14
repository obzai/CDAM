from cdam.myViT import MyVisionTransformer
from functools import partial
from torch import nn
from PIL import Image
from torchvision import transforms
import copy 
import matplotlib.colors as clr
import cmasher as cmr
import torch
import matplotlib.pyplot as plt
from cdam.classes import IMAGENET2012_CLASSES as imgnet_dict
from matplotlib.transforms import Bbox

# maps e.g. "bathing cap, swimming cap" -> "n02807133"
imgnet_dict_inv = {v: k for k, v in imgnet_dict.items()}
# maps e.g. 0 -> "tench, Tinca tinca"
idx2class = {i: j for i, j in enumerate(imgnet_dict.values())}
# maps e.g. "tench, Tinca tinca" -> 0
class2idx = {i: j for j, i in idx2class.items()}


def load_model(state_dict_path, num_classes):
    """Loads model from state dict and returns the model."""
    model = MyVisionTransformer(
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
    # check if state_dict_path is a string
    if isinstance(state_dict_path, str):
        model.load_state_dict(torch.load(state_dict_path), strict=True)
    # check if state_dict_path is a dict
    elif isinstance(state_dict_path, dict):
        model.load_state_dict(state_dict_path, strict=True)
    else:
        raise ValueError("state_dict_path must be a string or a dictionary")
    model.cuda()
    _ = model.eval()
    return model


transform = transforms.Compose(
    [
        # transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def load_img(img_path, img_size=(224, 224), patch_size=8):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize(img_size)
        original_img = copy.deepcopy(img)

    img = transform(img).cuda()

    # make image divisible by patch size
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)
    img.requires_grad = True
    return img, original_img


mycmap = clr.LinearSegmentedColormap.from_list(
    "Random gradient 1030",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%201030=0:00E3FF-37:4371AB-50:000000-63:8B5A44-100:FFA600
        (0.000, (0.000, 0.890, 1.000)),
        (0.370, (0.263, 0.443, 0.671)),
        (0.500, (0.000, 0.000, 0.000)),
        (0.630, (0.545, 0.353, 0.267)),
        (1.000, (1.000, 0.651, 0.000)),
    ),
)


def get_cmap(heatmap):
    """this returns a diverging colormap, such that 0 is at the center(black)"""
    if heatmap.min() > 0 and heatmap.max() > 0:
        bottom = 0.5
        top = 1.0
    elif heatmap.min() < 0 and heatmap.max() < 0:
        bottom = 0.0
        top = 0.5
    else:
        bottom = 0.5 - abs((heatmap.min() / abs(heatmap).max()) / 2)
        top = 0.5 + abs((heatmap.max() / abs(heatmap).max()) / 2)
    return cmr.get_sub_cmap(mycmap, bottom, top)


def plot_results(original, maps, savepath=None, plot_original=True, figsize=(9, 9), wspace=0.01):
    """Using matplotlib, plot the saliency maps in one row. If savepath is provided, save the plot to that path."""
    plt.figure(figsize=figsize, dpi=100)
    num_plots = len(maps)
    map_pos = 0
    if plot_original:
        num_plots += 1
        map_pos = 1
        plt.subplot(1, num_plots, 1)
        plt.imshow(original)
        plt.axis("off")
    for i, m in enumerate(maps):
        plt.subplot(1, num_plots, i + 1 + map_pos)
        # clip for higher contrast plots
        m = torch.clamp(
            m,
            min=torch.quantile(m, 0.001),
            max=torch.quantile(m, 0.999),
        )
        plt.imshow(m, cmap=get_cmap(m))
        plt.axis("off")
    plt.subplots_adjust(wspace=wspace, hspace=0)
    # save the plot to a file, cropped to only the image
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.show()

def plot_many_results(results_list, map_names, savepath, figsize, title_size):
    """Plot multiple results in a grid. Expects a list of dictionaries with keys 'original', 'maps', 'class'"""
    plt.figure(figsize=figsize)
    num_plots = 1 + len(results_list[0]["maps"])
    j = 1
    for row, res in enumerate(results_list):
        original = res["original"]
        maps = res["maps"]
        plt.subplot(len(results_list), num_plots, j)
        plt.imshow(original)
        plt.title(res["class"], fontsize=title_size)
        plt.axis("off")
        for i, m in enumerate(maps):
            m = torch.tensor(m)
            m = torch.clamp(
                m,
                min=torch.quantile(m, 0.015),
                max=torch.quantile(m, 0.995),
            )
            plt.subplot(len(results_list), num_plots, i + j + 1)
            plt.imshow(m, cmap=get_cmap(m))
            if row == 0:
                plt.title(map_names[i], fontsize=title_size)
            plt.axis("off")
        j = i + j + 2

    # Reduce the space between the subplots
    plt.subplots_adjust(wspace=0.0001, hspace=0.1)
    # plt.show()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(idx2class[cls_idx])
        if len(idx2class[cls_idx]) > max_str_len:
            max_str_len = len(idx2class[cls_idx])

    print("Top 5 classes:")
    for cls_idx in class_indices:
        output_string = "\t{} : {}".format(cls_idx, idx2class[cls_idx])
        output_string += " " * (max_str_len - len(idx2class[cls_idx])) + "\t\t"
        output_string += "value = {:.3f}\t prob = {:.1f}%".format(
            predictions[0, cls_idx], 100 * prob[0, cls_idx]
        )
        print(output_string)
