"""Simple 3-D regression and confidence losses for SCRegNet
================================================================
The losses follow DUSt3R's design but are adapted to the single-view
(world–coordinate) setting used by SCRegNet.

Key ideas
---------
1. *Pixel-wise 3-D regression*  (L2 on XYZ)
   L_reg = ‖p − p̂‖_2
2. *Confidence weighting*  (same as DUSt3R ConfLoss)
   L(p, p̂, τ) = τ · L_reg − α · log τ,  τ>0

The implementation keeps the API style of DUSt3R so models / trainers can
switch between the two without large code changes.
"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Tuple, Dict

import torch
import torch.nn as nn

__all__ = [
    "L21", "L21Loss", "Regr3D_World", "ConfLoss_World",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def Sum(*losses_and_masks):
    """Utility to aggregate (loss, mask) tuples.

    * If the first loss is pixel-wise (ndim>0) we return the full list to let
      the caller decide how to average.
    * Otherwise we assume scalar losses and simply sum them.
    """
    loss, _ = losses_and_masks[0]
    if loss.ndim > 0:
        return losses_and_masks  # pixel-level, keep as is
    total = loss
    for l, _ in losses_and_masks[1:]:
        total = total + l
    return total


# -----------------------------------------------------------------------------
# Basic criteria
# -----------------------------------------------------------------------------

class BaseCriterion(nn.Module):
    """Adds a `reduction` attribute like PyTorch losses."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """Generic L-p loss. Sub-classes implement :py:meth:`distance`."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert (
            a.shape == b.shape
            and a.ndim >= 2
            and 1 <= a.shape[-1] <= 3
        ), "Bad tensor shapes for LLoss"
        dist = self.distance(a, b)  # shape: a.shape[:-1]
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() else dist.new_zeros(())
        raise ValueError(f"Invalid reduction mode: {self.reduction}")

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501 – abstract
        raise NotImplementedError


class L21Loss(LLoss):
    """Euclidean distance between two 3-D points."""
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, dim=-1)


# Singleton for convenience (mimic DUSt3R)
L21 = L21Loss()


# -----------------------------------------------------------------------------
# Composable loss wrappers (borrowed from DUSt3R)
# -----------------------------------------------------------------------------

class Criterion(nn.Module):
    """Wrap a **pixel-level** criterion so we can later chain with MultiLoss."""
    def __init__(self, criterion: BaseCriterion):
        super().__init__()
        assert isinstance(criterion, BaseCriterion)
        self.criterion = copy(criterion)

    # Pretty-print ----------------------------------------------------------------
    def get_name(self) -> str:
        return f"{type(self).__name__}({self.criterion})"

    # Utilities -------------------------------------------------------------------
    def with_reduction(self, mode: str = "none") -> "Criterion":
        res: Criterion = deepcopy(self)
        cur: Criterion | None = res
        while cur:
            cur.criterion.reduction = mode
            cur = cur._loss2  # type: ignore[attr-defined]
        return res


class MultiLoss(nn.Module):
    """Chainable loss container:  * + α*  semantics like in DUSt3R."""
    def __init__(self) -> None:
        super().__init__()
        self._alpha: float = 1.0
        self._loss2: MultiLoss | None = None

    # These two must be implemented by subclasses
    def compute_loss(self, *args, **kwargs):  # noqa: ANN001
        raise NotImplementedError
    def get_name(self) -> str:  # noqa: D401
        raise NotImplementedError

    # Operator overloads
    def __mul__(self, alpha: float):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__

    def __add__(self, other: "MultiLoss"):
        assert isinstance(other, MultiLoss)
        res = cur = copy(self)
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = other
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    # Main forward ----------------------------------------------------------------
    def forward(self, *args, **kwargs):  # noqa: ANN001
        loss, details = self.compute_loss(*args, **kwargs)
        loss = loss * self._alpha
        if self._loss2 is not None:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2
        return loss, details


# -----------------------------------------------------------------------------
# Single-view regression losses (world frame)
# -----------------------------------------------------------------------------

class Regr3D_World(Criterion, MultiLoss):
    """Pixel-level L2 loss between predicted & GT 3-D points in the **same world
    coordinate frame**.
    Expected sample dicts:
    * **gt**   – keys `pts3d` (B×H×W×3), `valid_mask` (B×H×W)
    * **pred** – key  `pts3d` (B×H×W×3)
    """

    def __init__(self, criterion: BaseCriterion = L21):
        super().__init__(criterion)

    # ------------------------------------------------------------------
    def compute_loss(self, gt: Dict, pred: Dict, **kw):  # noqa: ANN001
        gt_pts: torch.Tensor = gt["pts3d"]
        pr_pts: torch.Tensor = pred["pts3d"]
        valid: torch.Tensor = gt["valid_mask"].clone()

        l = self.criterion(pr_pts[valid], gt_pts[valid])
        self_name = type(self).__name__
        details = {self_name: float(l.mean())}
        return (l, valid), details

    # Override -------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Directly call :py:meth:`compute_loss` so that the list output is **not**
        multiplied by *alpha* inside :class:`MultiLoss.forward`, which would
        otherwise raise ``TypeError: can't multiply sequence by non-int``.
        """
        return self.compute_loss(*args, **kwargs)


class ConfLoss_World(MultiLoss):
    """Confidence-aware extension of :class:`Regr3D_World`.

    L = τ * L_reg  − α * log τ
    where τ=predicted confidence (>0).
    """

    def __init__(self, pixel_loss: Regr3D_World, alpha: float = 1.0):
        super().__init__()
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    # Pretty-print ---------------------------------------------------------------
    def get_name(self):
        return f"ConfLoss_World({self.pixel_loss})"

    # ---------------------------------------------------------------------------
    @staticmethod
    def _conf_and_log(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.log(x)

    def compute_loss(self, gt: Dict, pred: Dict, **kw):  # noqa: ANN001
        (loss_pix, mask), details = self.pixel_loss(gt, pred, **kw)
        if loss_pix.numel() == 0:
            print("NO VALID POINTS", force=True)

        conf, log_conf = self._conf_and_log(pred["conf"][mask])
        conf_loss = loss_pix * conf - self.alpha * log_conf
        conf_loss = conf_loss.mean() if conf_loss.numel() else loss_pix.new_zeros(())
        return conf_loss, {"conf_loss": float(conf_loss), **details}
