"""SCRegNet prediction head factory.
Copypasta from dust3r but routed through local package so that SCRegNet can work
without importing from dust3r outside the sub-module.
"""
from .linear_head import LinearPts3d
import importlib


def head_factory(head_type: str, output_mode: str, net, has_conf: bool = False):
    """Return a prediction head instance.

    Parameters
    ----------
    head_type: str
        Either ``'linear'`` or ``'dpt'``.
    output_mode: str
        Currently only ``'pts3d'`` is supported.
    net: CroCoNet-like instance
        Network that owns the decoder whose features will feed the head.
    has_conf: bool, default False
        If True, the head should also predict a confidence channel.
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    if head_type == 'dpt' and output_mode == 'pts3d':
        dpt_module = importlib.import_module('.dpt_head', package=__name__)
        return dpt_module.create_dpt_head(net, has_conf=has_conf)

    raise NotImplementedError(f"Unsupported combination {head_type=} {output_mode=}")
