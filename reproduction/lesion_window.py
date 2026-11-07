"""Learn a three-channel CT contrast adjustment from current-slice predicted anatomy."""
import torch
from torch import nn
from torch.nn import functional as F


def anatomy_windows(image, guide):
    support = ((guide > 0).sum(1, keepdim=True) >= 2).to(image.dtype)
    core = -F.max_pool2d(-support, 3, stride=1, padding=1) > 0
    ring = (F.max_pool2d(support, 13, stride=1, padding=6) > 0) & (F.max_pool2d(support, 3, stride=1, padding=1) == 0)
    quantiles = image.new_tensor([0.1, 0.5, 0.9])
    def measurements(region):
        values = image.masked_fill(~region, torch.nan).flatten(1)
        return torch.nanquantile(values, quantiles, dim=1).nan_to_num()
    low, middle, high = measurements(core)
    background, surrounding, _ = measurements(ring)
    span = (high - background).clamp_min(12/255)
    midpoint = (surrounding + middle) / 2
    left = torch.stack((background, midpoint-span/4, low), dim=1)
    right = torch.stack((high, midpoint+span/4, high), dim=1)
    width = (right-left).clamp_min(12/255)
    lower = (right+left-width)/2
    channels = ((image-lower[:,:,None,None])/width[:,:,None,None]).clamp(0,1)
    valid = (core.flatten(1).sum(1)>=8) & (ring.flatten(1).sum(1)>=8) & (middle-surrounding>4/255)
    return channels, valid


class LesionWindow(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(3))
        self.register_buffer("mean", torch.tensor([123.675,116.28,103.53])[None,:,None,None])
        self.register_buffer("std", torch.tensor([58.395,57.12,57.375])[None,:,None,None])

    def forward(self, original, image, guide):
        windows, valid = anatomy_windows(image.float(), guide)
        windows = F.interpolate(windows, original.shape[-2:], mode="bilinear", align_corners=False)
        alternate = (255*windows-self.mean)/self.std
        strength = 0.5*torch.tanh(2*self.gates)[None,:,None,None]
        return original + valid[:,None,None,None]*strength*(alternate-original)
