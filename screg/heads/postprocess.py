"""Postprocess utilities copied from dust3r to convert head outputs to dict.
"""
import torch

def postprocess(out, depth_mode=None, conf_mode=None):
    """Extract 3D points/confidence from head output tensor.

    Parameters
    ----------
    out : Tensor, shape (B, C, H, W)
        Raw output of head before reshaping.
    depth_mode, conf_mode : tuple
        Modes as described in dust3r paper.
    """
    # Expect out shape (B,C,H,W) with C = 3 (+1 conf if present).
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C
    # First 3 channels encode (x,y,z) direction/depth per pixel
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    # Optional confidence channel if present and requested
    if conf_mode is not None and fmap.shape[-1] > 3:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def reg_dense_depth(xyz, mode):
    mode, vmin, vmax = mode
    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz
        return xyz.clip(min=vmin, max=vmax)

    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clamp(min=1e-8)

    if mode == 'square':
        return xyz * d.square()
    if mode == 'exp':
        return xyz * torch.expm1(d)
    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clamp(max=vmax - vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')
