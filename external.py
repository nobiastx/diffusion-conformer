import torch
from torch import Tensor
from typing import Dict

class pair_repulsion:
    """External force designed to prevent atom overlaps"""

    def __init__(self, strength: float, clip: float=0.7):
        self.strength = strength
        self.clip2 = clip*clip

    def __call__(self, x: Dict[str,Tensor], sigma: Tensor) -> Tensor:
        coords = x['coords']
        pairs = x['pairs']

        dr = torch.index_select(coords,0,pairs[:,0]) - torch.index_select(coords,0,pairs[:,1])
        dl2 = torch.square(dr).sum(-1)
        dl = torch.sqrt(dl2.clamp(min=1E-12))
        dh = dr/dl.unsqueeze(-1)

        U = self.strength * torch.pow(dl2.clamp(min=self.clip2),-5)
        answer = torch.zeros(coords.shape, dtype=coords.dtype, device=coords.device)
        answer.index_add_(0, pairs[:,0], U.unsqueeze(-1)*dh, alpha= 0.5)
        answer.index_add_(0, pairs[:,1], U.unsqueeze(-1)*dh, alpha=-0.5)
        return answer
