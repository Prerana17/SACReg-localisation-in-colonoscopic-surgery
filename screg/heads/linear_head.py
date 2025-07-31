"""Linear prediction head copied from dust3r for SCRegNet use."""
import torch.nn as nn
import torch.nn.functional as F
from .postprocess import postprocess  # relative import within screg package


class LinearPts3d(nn.Module):
    """Linear head for SCRegNet to produce 3D points (and optionally confidence)."""

    def __init__(self, net, has_conf: bool = False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = getattr(net, 'depth_mode', ('exp', -float('inf'), float('inf')))
        self.conf_mode = getattr(net, 'conf_mode', ('exp', 1, float('inf')))
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf) * self.patch_size ** 2)

    def setup(self, croconet):  # kept for API compatibility
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1] if isinstance(decout, (list, tuple)) else decout  # B,S,D
        B, S, D = tokens.shape
        feat = self.proj(tokens)  # B,S,(3+conf)*p^2
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3(+conf),H,W
        return postprocess(feat, self.depth_mode, self.conf_mode)
