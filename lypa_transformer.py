
# Calculates Lyapunov exponents to estimate edge of chaos initialization. 
# This analysis is based on:

# Geometric Dynamics of Signal Propagation Predict Trainability of Transformers
# Cowsik, Nebabu, Qi, Ganguli (arXiv:2403.02579)

# It backs the success of standard initialization schemes that choose α∼1/√L.

from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@dataclass
class H:
    d_model: int = 64
    d_mlp: int = 4*64
    n_tokens: int = 256
    alpha_A: float = 0.5     # residual strength for attention branch
    alpha_M: float = 0.5     # residual strength for MLP branch
    sigma_A: float = 1.0     # controls Q,K scale so Var(Q^T K) ~ σ_A^2/d
    sigma_w: float = 2.0     # MLP first-layer scale
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float32
torch.set_default_dtype(dtype)
EPS = torch.finfo(dtype).eps

class AcausalSelfAttention(nn.Module):
    """ 
    """
    def __init__(self, d_model: int, sigma_A: float):
        super().__init__()
        self.d = d_model
        qk_std = math.sqrt(sigma_A/d_model)   # Assumption 3.3
        # qk_std = (sigma_A**0.5)/math.sqrt(self.d)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.query.weight, 0.0, qk_std)
        nn.init.normal_(self.key.weight, 0.0, qk_std)
        nn.init.normal_(self.value.weight, 0.0, 1.0/math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        q = self.query(x)
        logits = (q @ k.T) * self.d**-0.5 
        A = F.softmax(logits, dim=-1)
        out = self.value(A@x)
        return out

class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, sigma_w: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=False)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=False)
        # use pre-activation variance 
        nn.init.normal_(self.fc1.weight, 0.0, sigma_w/math.sqrt(d_model))
        # use readout unit variance 
        nn.init.normal_(self.fc2.weight, 0.0, 1/math.sqrt(d_mlp))

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

class TransformerBlock(nn.Module):
    """
    single-head self-attention block followed by tokenwise 2-layer perceptron
    """
    def __init__(self, h: H):
        super().__init__()
        self.h = h
        self.attn = AcausalSelfAttention(h.d_model, h.sigma_A)
        self.mlp  = MLP(h.d_model, h.d_mlp, h.sigma_w)

    def layernorm(self, x: torch.Tensor) -> torch.Tensor:
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
        out = x / nrm * math.sqrt(self.h.d_model)
        # (n_tokens, d_model)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equation 4 (Figure 1)
        a_comp = math.sqrt(max(0.0, 1.0 - self.h.alpha_A**2))
        m_comp = math.sqrt(max(0.0, 1.0 - self.h.alpha_M**2))
        x = a_comp*x + self.h.alpha_A*self.attn(self.layernorm(x))
        x = m_comp*x + self.h.alpha_M*self.mlp(self.layernorm(x))
        return x

# @torch.no_grad()
# def make_tokens(n: int, d: int, device) -> torch.Tensor:
#     """ 
#     """
#     x = torch.randn(n, d, device=device)
#     x = x / x.norm(dim=1, keepdim=True) * math.sqrt(d)
#     return x

@torch.no_grad()
def make_tokens(n: int, d: int, rho: float = 0.99, device="cpu"):
    """
    E||X_i||^2 = d  and  E[X_i · X_j] = rho * d  (i != j).
    """
    g  = torch.randn(d, device=device)
    Z  = torch.randn(n, d, device=device)
    X  = math.sqrt(rho) * g.unsqueeze(0) + math.sqrt(1.0 - rho) * Z
    return X

def qp_stats(x: torch.Tensor) -> Tuple[float, float]:
    """ 
    """
    C = x @ x.T
    n = C.shape[0]
    diag = torch.diagonal(C)
    # diagonal average
    q = diag.mean().item()
    # off-diagonal average
    p = ((C.sum() - diag.sum()) / (n*(n-1))).item()
    # print(f"q={q} and p={p}")
    return max(EPS, q), max(EPS, p)

@torch.no_grad()
def angle_lyapunov_exponent(h: H, seeds: int = 1) -> float:
    """
    λ_a ≈ log((1 - p'/q') / (1 - p/q))
    Section 4.1
    """
    vals = []
    for s in range(seeds):
        torch.manual_seed(s)
        block = TransformerBlock(h).to(h.device).eval()
        x0 = make_tokens(h.n_tokens, h.d_model, device=h.device)
        q, p = qp_stats(x0)
        x1 = block(x0)
        q1, p1 = qp_stats(x1)
        r0 = max(EPS, 1.0 - p/q)
        r1 = max(EPS, 1.0 - p1/q1)
        vals.append(math.log(r1/r0))
    return sum(vals)/len(vals)

def gradient_lyapunov_exponent(h: H, depth: int = 16, seeds: int = 1) -> float:
    """
    λ_g ≈ (1/L) * log || d(X_L · R) / dX_0 ||_F^2
    Sections 3.4 & 4.2
    """
    vals = []
    for s in range(seeds):
        torch.manual_seed(s)
        blocks = nn.ModuleList([TransformerBlock(h).to(h.device).eval() for _ in range(depth)])
        X0 = make_tokens(h.n_tokens, h.d_model, rho=0.99, device=h.device).requires_grad_(True)
        Xt = X0
        for b in blocks:
            Xt = b(Xt)
        R = torch.randn_like(Xt)
        scalar = (Xt * R).sum()
        (dX,) = torch.autograd.grad(scalar, X0, retain_graph=False, create_graph=False, allow_unused=False)
        v = (dX**2).sum().item()
        vals.append(math.log(max(v, EPS)) / depth)
    return sum(vals)/len(vals)

if __name__ == "__main__":
    # grid search for edge of chaos where \lamda_a is 0
    res = 25
    Z = torch.zeros(res, res, dtype=dtype)
    for i,sw in enumerate(torch.linspace(1, 4, res)):
        for j,a in enumerate(torch.linspace(0, 1, res)):
            h = H(alpha_A=a, alpha_M=a, sigma_w=sw, sigma_A=0.6)
            Z[i,j] = angle_lyapunov_exponent(h, seeds=5)

    # Figure 4 (top-right)
    vlim = torch.abs(Z).max()
    cmap = plt.get_cmap("bwr")  # blue-white-red
    fig, ax = plt.subplots(figsize=(5.5,4.5), dpi=130)
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap=cmap, norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim))
    ax.set_xlabel(r'$\alpha = \alpha_A = \alpha_M$')
    ax.set_ylabel(r'$\sigma_w$')
    fig.colorbar(im, ax=ax, label=r'$\lambda_a$')
    plt.tight_layout()
    plt.show()
