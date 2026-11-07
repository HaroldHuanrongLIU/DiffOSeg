"""Low-rank query/value updates in a frozen original SAM image prefix."""
import torch
from torch import nn


class LowRankQv(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original=original.requires_grad_(False)
        width=original.in_features
        self.query_down=nn.Linear(width,rank,bias=False)
        self.value_down=nn.Linear(width,rank,bias=False)
        self.query_up=nn.Linear(rank,width,bias=False)
        self.value_up=nn.Linear(rank,width,bias=False)
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.value_up.weight)
        self.to(original.weight.device)

    def forward(self,x):
        q,k,v=self.original(x).chunk(3,dim=-1)
        return torch.cat((q+self.query_up(self.query_down(x)),k,v+self.value_up(self.value_down(x))),dim=-1)


class ImagePrefix(nn.Module):
    def __init__(self,encoder,adapt_attention):
        super().__init__()
        self.patch=encoder.patch_embed
        self.position=encoder.pos_embed
        self.blocks=nn.ModuleList(encoder.blocks[:-4])
        if adapt_attention:
            # The added CPU-initialized matrices do not advance the paired augmentation RNG.
            with torch.random.fork_rng(devices=[]):
                for block in self.blocks:
                    block.attn.qkv=LowRankQv(block.attn.qkv)

    def forward(self,image):
        x=self.patch(image)
        if self.position is not None:
            x=x+self.position
        for block in self.blocks:
            x=block(x)
        return x
