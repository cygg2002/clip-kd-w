import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _kl_div_from_probs(p_t: torch.Tensor, log_p_s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(p_t || p_s) = sum p_t * (log p_t - log p_s)
    p_t: [B, K] probabilities (should sum to 1)
    log_p_s: [B, K] log probabilities
    """
    p_t = p_t.clamp_min(eps)
    log_p_t = torch.log(p_t)
    kl = torch.sum(p_t * (log_p_t - log_p_s), dim=-1)
    return kl.mean()


def _kd_kl_logits(z_t: torch.Tensor, z_s: torch.Tensor, tau: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Distillation KL on logits with temperature tau.
    z_t, z_s: [B, K]
    """
    p_t = F.softmax(z_t / tau, dim=-1)
    log_p_s = F.log_softmax(z_s / tau, dim=-1)
    return (tau * tau) * _kl_div_from_probs(p_t, log_p_s, eps=eps)


def _scatter_sum_probs(p_sp: torch.Tensor, sp_to_group: torch.Tensor, n_group: int) -> torch.Tensor:
    """
    Aggregate species probabilities to group probabilities by scatter_add.
    p_sp: [B, N_sp]
    sp_to_group: [N_sp] values in [0, n_group-1]
    returns p_group: [B, n_group]
    """
    B, N = p_sp.shape
    device = p_sp.device
    idx = sp_to_group.view(1, -1).expand(B, -1)  # [B, N_sp]
    out = torch.zeros((B, n_group), device=device, dtype=p_sp.dtype)
    out.scatter_add_(1, idx, p_sp)
    return out


def _expand_uniform(p_group: torch.Tensor, sp_to_group: torch.Tensor, group_size: torch.Tensor) -> torch.Tensor:
    """
    Uniformly expand group distribution back to species space.
    p_group: [B, n_group]
    sp_to_group: [N_sp]
    group_size: [n_group] (>=1)
    returns q_sp: [B, N_sp]
    q_sp[:, i] = p_group[:, sp_to_group[i]] / group_size[sp_to_group[i]]
    """
    B, _ = p_group.shape
    device = p_group.device
    idx = sp_to_group.view(1, -1).expand(B, -1)                     # [B, N_sp]
    denom = group_size[sp_to_group].view(1, -1).to(device=device)   # [1, N_sp]
    q = p_group.gather(1, idx) / denom
    return q


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


# -----------------------------
# Config
# -----------------------------
@dataclass
class TaxoDistillConfig:
    # Which levels to use (ordered from fine -> coarse)
    levels: Tuple[str, ...] = ("species", "genus", "family", "order", "class")

    # Temperature for KD
    tau: float = 4.0

    # Hierarchical KD weights per level
    lambda_level: Optional[Dict[str, float]] = None  # if None, set defaults

    # Distance-calibrated KD settings (species-level only)
    use_distance_kd: bool = True
    mu_dist: float = 0.2        # weight for distance KD (you can schedule externally)
    gamma: float = 1.0          # distance decay. larger => less smoothing
    # Which coarser levels used to smooth species distribution (distance 1..)
    smoothing_levels: Tuple[str, ...] = ("genus", "family", "order", "class")

    # Representation KD
    use_repr_kd: bool = True
    beta_repr: float = 0.5
    repr_loss: str = "cosine"   # "cosine" or "mse"

    eps: float = 1e-12

 
# -----------------------------
# Loss module
# -----------------------------
class TaxoDistillLoss(nn.Module):
    """
    Class-level taxonomy distillation for CLIP-like models:
      1) Hierarchical KD: KL at each taxonomy level logits (species/genus/family/order/class).
      2) Distance-calibrated KD: smooth teacher species distribution using taxonomy proximity,
         then KL to student species distribution.
      3) Representation KD: align teacher/student image embeddings (cosine or MSE), with optional projection.

    Expected inputs:
      - student image features:  [B, D_s] (unnormalized ok)
      - teacher image features:  [B, D_t] (unnormalized ok)
      - student logit_scale: scalar tensor (or float)
      - teacher logit_scale: scalar tensor (or float)
      - text vocab embeddings dict per level for student and teacher:
          s_text[level]: [N_level, D_s]
          t_text[level]: [N_level, D_t]
        (must be L2-normalized or we normalize inside)
      - taxonomy mappings for distance smoothing:
          sp_to_gen: [N_sp] long
          sp_to_fam: [N_sp] long
          sp_to_ord: [N_sp] long
          sp_to_cls: [N_sp] long
          gen_size: [N_gen] float/long (>=1)
          fam_size: [N_fam] ...
          ord_size: [N_ord] ...
          cls_size: [N_cls] ...
    """

    def __init__(
        self,
        cfg: TaxoDistillConfig,
        s_embed_dim: int,
        t_embed_dim: int,
    ):
        super().__init__()
        self.cfg = cfg

        # Default lambda schedule (fine -> coarse)
        if self.cfg.lambda_level is None:
            self.cfg.lambda_level = {
                "species": 1.0,
                "genus":   0.4,
                "family":  0.2,
                "order":   0.1,
                "class":   0.05
                # "species": 0.0,
                # "genus":   0.0,
                # "family":  0.0,
                # "order":   0.0,
                # "class":   0.0
            }

        # Optional projection if dims differ for representation KD
        self.need_proj = (s_embed_dim != t_embed_dim)
        if self.need_proj:
            self.img_proj = nn.Linear(s_embed_dim, t_embed_dim)

        # Buffers for taxonomy mappings (set via set_taxonomy)
        self.register_buffer("sp_to_gen", None, persistent=False)
        self.register_buffer("sp_to_fam", None, persistent=False)
        self.register_buffer("sp_to_ord", None, persistent=False)
        self.register_buffer("sp_to_cls", None, persistent=False)

        self.register_buffer("gen_size", None, persistent=False)
        self.register_buffer("fam_size", None, persistent=False)
        self.register_buffer("ord_size", None, persistent=False)
        self.register_buffer("cls_size", None, persistent=False)

        # Optional cached vocab embeddings (set via set_text_embeddings)
        # (persistent=False avoids blowing up checkpoints unless you want them saved)
        self._s_text = nn.ModuleDict()  # store as parameters? no, but ModuleDict holds Tensors poorly
        self._t_text = nn.ModuleDict()
        # We will store cached embeddings as buffers in dict-like manner:
        self._cached_s_text: Dict[str, torch.Tensor] = {}
        self._cached_t_text: Dict[str, torch.Tensor] = {}

    # ---------- Setup helpers ----------
    @torch.no_grad()
    def set_taxonomy(
        self,
        sp_to_gen: torch.Tensor,
        sp_to_fam: torch.Tensor,
        sp_to_ord: torch.Tensor,
        sp_to_cls: torch.Tensor,
        gen_size: torch.Tensor,
        fam_size: torch.Tensor,
        ord_size: torch.Tensor,
        cls_size: torch.Tensor,
    ):
        """
        Set taxonomy mapping buffers.
        All tensors should be on CPU or GPU; we will register as buffers.
        """
        self.sp_to_gen = sp_to_gen.long()
        self.sp_to_fam = sp_to_fam.long()
        self.sp_to_ord = sp_to_ord.long()
        self.sp_to_cls = sp_to_cls.long()

        self.gen_size = gen_size.float().clamp_min(1.0)
        self.fam_size = fam_size.float().clamp_min(1.0)
        self.ord_size = ord_size.float().clamp_min(1.0)
        self.cls_size = cls_size.float().clamp_min(1.0)

    @torch.no_grad()
    def set_text_embeddings(
        self,
        s_text: Dict[str, torch.Tensor],
        t_text: Dict[str, torch.Tensor],
        normalize: bool = True,
    ):
        """
        Cache vocab embeddings inside the loss module (optional but convenient).
        s_text[level]: [N_level, D_s]
        t_text[level]: [N_level, D_t]
        """
        self._cached_s_text = {}
        self._cached_t_text = {}
        for lvl, emb in s_text.items():
            self._cached_s_text[lvl] = _normalize_rows(emb) if normalize else emb
        for lvl, emb in t_text.items():
            self._cached_t_text[lvl] = _normalize_rows(emb) if normalize else emb

    # ---------- Core computations ----------
    def _get_text(self, s_text: Optional[Dict[str, torch.Tensor]], t_text: Optional[Dict[str, torch.Tensor]]):
        s_out = s_text if s_text is not None else self._cached_s_text
        t_out = t_text if t_text is not None else self._cached_t_text
        if not s_out or not t_out:
            raise ValueError("Text vocab embeddings not provided and not cached. Call set_text_embeddings(...) or pass s_text/t_text to forward.")
        return s_out, t_out

    def _compute_level_logits(
        self,
        img_feat: torch.Tensor,          # [B, D]
        text_emb: torch.Tensor,          # [N, D]
        logit_scale: torch.Tensor,       # scalar
    ) -> torch.Tensor:
        """
        CLIP-like logits: scale * (normalize(img) @ normalize(text).T)
        """
        img = _normalize_rows(img_feat, eps=self.cfg.eps)
        txt = _normalize_rows(text_emb, eps=self.cfg.eps)
        return logit_scale * (img @ txt.t())

    def _hierarchical_kd(
        self,
        zT: Dict[str, torch.Tensor],
        zS: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Per-level KL distillation.
        """
        tau = self.cfg.tau
        parts = {}
        total = torch.zeros((), device=next(iter(zS.values())).device)
        for lvl in self.cfg.levels:
            if lvl not in zT or lvl not in zS:
                continue
            l = _kd_kl_logits(zT[lvl].detach(), zS[lvl], tau=tau, eps=self.cfg.eps)
            w = float(self.cfg.lambda_level.get(lvl, 0.0))
            parts[f"kd_{lvl}"] = l
            total = total + w * l
        parts["kd_hier_total"] = total
        return total, parts

    def _distance_calibrated_kd_species(
        self,
        zT_sp: torch.Tensor,     # [B, N_sp]
        zS_sp: torch.Tensor,     # [B, N_sp]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Distance-calibrated KD on species distribution:
          pT_sp -> smooth with genus/family/order/class -> KL to student species.
        """
        device = zS_sp.device
        tau = self.cfg.tau

        # teacher species probs
        pT_sp = F.softmax(zT_sp.detach() / tau, dim=-1)  # [B, N_sp]
        log_pS = F.log_softmax(zS_sp / tau, dim=-1)

        # Safety checks: taxonomy buffers must exist
        if self.sp_to_gen is None or self.gen_size is None:
            raise ValueError("Taxonomy buffers not set. Call set_taxonomy(...) before using distance KD.")

        B, N_sp = pT_sp.shape
        parts = {}

        # Build smoothing sources: d=0 uses original pT_sp
        sources: List[Tuple[int, torch.Tensor]] = [(0, pT_sp)]

        # Helper to add one smoothing level
        def add_level(dist_d: int, level: str):
            if level == "genus":
                sp2g = self.sp_to_gen.to(device=device)
                p_g = _scatter_sum_probs(pT_sp, sp2g, int(self.gen_size.numel()))
                q = _expand_uniform(p_g, sp2g, self.gen_size.to(device=device))
                sources.append((dist_d, q))
            elif level == "family":
                sp2f = self.sp_to_fam.to(device=device)
                p_f = _scatter_sum_probs(pT_sp, sp2f, int(self.fam_size.numel()))
                q = _expand_uniform(p_f, sp2f, self.fam_size.to(device=device))
                sources.append((dist_d, q))
            elif level == "order":
                sp2o = self.sp_to_ord.to(device=device)
                p_o = _scatter_sum_probs(pT_sp, sp2o, int(self.ord_size.numel()))
                q = _expand_uniform(p_o, sp2o, self.ord_size.to(device=device))
                sources.append((dist_d, q))
            elif level == "class":
                sp2c = self.sp_to_cls.to(device=device)
                p_c = _scatter_sum_probs(pT_sp, sp2c, int(self.cls_size.numel()))
                q = _expand_uniform(p_c, sp2c, self.cls_size.to(device=device))
                sources.append((dist_d, q))
            else:
                raise ValueError(f"Unknown smoothing level: {level}")

        # Add smoothing levels with distance 1..k in the given order
        d = 1
        for lvl in self.cfg.smoothing_levels:
            add_level(d, lvl)
            d += 1

        # Compute weights w_d ∝ exp(-gamma d)
        gamma = self.cfg.gamma
        ds = torch.tensor([dd for dd, _ in sources], device=device, dtype=torch.float32)  # [M]
        w = torch.softmax(-gamma * ds, dim=0)  # [M]
        parts["dist_weights"] = w.detach()

        # Weighted mixture
        pT_smooth = torch.zeros_like(pT_sp)
        for i, (_, q) in enumerate(sources):
            pT_smooth = pT_smooth + w[i] * q

        # KL(pT_smooth || pS)
        L_dist = (tau * tau) * _kl_div_from_probs(pT_smooth, log_pS, eps=self.cfg.eps)
        parts["kd_dist_species"] = L_dist
        return L_dist, parts

    def _repr_kd(
        self,
        s_img_feat: torch.Tensor,  # [B, D_s]
        t_img_feat: torch.Tensor,  # [B, D_t]
    ) -> torch.Tensor:
        """
        Representation KD between image embeddings (teacher vs student).
        """
        if self.need_proj:
            s = self.img_proj(s_img_feat)
        else:
            s = s_img_feat
        s = _normalize_rows(s, eps=self.cfg.eps)
        t = _normalize_rows(t_img_feat.detach(), eps=self.cfg.eps)

        if self.cfg.repr_loss == "cosine":
            # 1 - cosine similarity
            return (1.0 - torch.sum(s * t, dim=-1)).mean()
        elif self.cfg.repr_loss == "mse":
            return F.mse_loss(s, t)
        else:
            raise ValueError(f"Unknown repr_loss: {self.cfg.repr_loss}")

    # ---------- Forward ----------
    def forward(
        self,
        s_image_features: torch.Tensor,      # [B, D_s]
        t_image_features: torch.Tensor,      # [B, D_t]
        s_logit_scale: torch.Tensor,         # scalar
        t_logit_scale: torch.Tensor,         # scalar
        s_text: Optional[Dict[str, torch.Tensor]] = None,
        t_text: Optional[Dict[str, torch.Tensor]] = None,
        mu_dist: Optional[float] = None,     # allow external scheduling
        return_parts: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns (total_loss, parts_dict).
        parts_dict includes individual components for logging.
        """
        device = s_image_features.device
        s_text_dict, t_text_dict = self._get_text(s_text, t_text)

        # Compute logits per level for teacher and student
        zT: Dict[str, torch.Tensor] = {}
        zS: Dict[str, torch.Tensor] = {}

        for lvl in self.cfg.levels:
            if lvl not in s_text_dict or lvl not in t_text_dict:
                continue
            zT[lvl] = self._compute_level_logits(t_image_features, t_text_dict[lvl].to(device=device), t_logit_scale)
            zS[lvl] = self._compute_level_logits(s_image_features, s_text_dict[lvl].to(device=device), s_logit_scale)

        # 1) Hierarchical KD
        L_hier, parts_hier = self._hierarchical_kd(zT, zS)

        # 2) Distance-calibrated KD (species-level)
        parts_dist = {}
        if self.cfg.use_distance_kd and ("species" in zT) and ("species" in zS):
            mu = self.cfg.mu_dist if (mu_dist is None) else float(mu_dist)
            if mu > 0:
                L_dist, parts_dist = self._distance_calibrated_kd_species(zT["species"], zS["species"])
            else:
                L_dist = torch.zeros((), device=device)
            L_dist = mu * L_dist
        else:
            L_dist = torch.zeros((), device=device)

        # 3) Representation KD (image embedding)
        if self.cfg.use_repr_kd:
            L_repr = self.cfg.beta_repr * self._repr_kd(s_image_features, t_image_features)
        else:
            L_repr = torch.zeros((), device=device)

        total = L_hier + L_dist + L_repr

        if not return_parts:
            return total, {}

        parts = {}
        parts.update(parts_hier)
        parts.update(parts_dist)
        parts["loss_hier"] = L_hier.detach()
        parts["loss_dist"] = L_dist.detach()
        parts["loss_repr"] = L_repr.detach()
        parts["loss_total_taxo"] = total.detach()
        return total, parts



