"""DPT-style prediction head copied from dust3r for SCRegNet.
Only import paths changed to stay within the `screg` package.
"""
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from .postprocess import postprocess
from croco.models.dpt_block import DPTOutputAdapter  # corrected import


class DPTOutputAdapterFix(DPTOutputAdapter):
    """Adopt croco's DPTOutputAdapter for SCRegNet (remove duplicate weights)."""

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # Remove duplicated weights that exist in croco's implementation
        for name in ("act_1_postprocess", "act_2_postprocess", "act_3_postprocess", "act_4_postprocess"):
            if hasattr(self, name):
                delattr(self, name)

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) first"
        H, W = image_size if image_size is not None else self.image_size
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        layers = [encoder_tokens[idx] for idx in self.hooks]
        layers = [self.adapt_tokens(l) for l in layers]
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
        layers = [self.act_postprocess[i](l) for i, l in enumerate(layers)]
        layers = [self.scratch.layer_rn[i](l) for i, l in enumerate(layers)]

        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)
        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT head producing dense 3D points (+ confidence)."""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super().__init__()
        self.return_all_layers = True
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapterFix(**dpt_args)
        init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf: bool = False):
    """Factory helper to build DPT head given SCRegNet params."""
    assert net.dec_depth > 9, "DPT head expects deep decoder"
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')
