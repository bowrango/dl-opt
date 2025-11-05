
# Computes the edge of chaos for randomly initialized transformers. 
# It backs the success of standard initialization schemes that choose α∼1/√L

# Geometric Dynamics of Signal Propagation Predict Trainability of Transformers
# (https://arxiv.org/pdf/2403.02579)

from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

@dataclass
class Config:
    d: int = 64
    d_mlp: int = 64
    n_tokens: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # hyperparameters:
    alpha_A: float = 1.0     # ATTN gain
    alpha_M: float = 1.0     # MLP gain
    sigma_A: float = 1.0     # ATTN variance
    sigma_w: float = 1.0     # MLP variance

dtype = torch.float32
torch.set_default_dtype(dtype)
EPS = torch.finfo(dtype).eps

class Head(nn.Module):
    """ 
    non-casual attention head
    """
    def __init__(self, d: int, sigma_A: float):
        super().__init__()
        self.d = d
        # Assumption 3.3: var[Q^T K] ~ σ_A^2/d
        qk_std = math.sqrt(sigma_A/d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.query.weight, 0.0, qk_std)
        nn.init.normal_(self.key.weight, 0.0, qk_std)
        nn.init.normal_(self.value.weight, 0.0, 1.0/math.sqrt(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        q = self.query(x)
        logits = (q @ k.T) * self.d**-0.5 
        A = logits.softmax(dim=-1)
        out = self.value(A @ x)
        # (n_tokens, d)
        return out

class MLP(nn.Module):
    def __init__(self, d: int, d_mlp: int, sigma_w: float):
        super().__init__()
        self.fc1 = nn.Linear(d, d_mlp, bias=False)
        self.fc2 = nn.Linear(d_mlp, d, bias=False)
        nn.init.normal_(self.fc1.weight, 0.0, sigma_w/math.sqrt(d))
        nn.init.normal_(self.fc2.weight, 0.0, sigma_w/math.sqrt(d))

    def forward(self, x) -> torch.Tensor:
        out = self.fc2(torch.tanh(self.fc1(x)))
        # (n_tokens, d)
        return out

class TransformerBlock(nn.Module):
    """
    single-head self-attention block followed by tokenwise 2-layer perceptron
    """
    def __init__(self, h: Config):
        super().__init__()
        self.h = h
        self.attn = Head(h.d, h.sigma_A)
        self.mlp  = MLP(h.d, h.d_mlp, h.sigma_w)

    def layernorm(self, x: torch.Tensor) -> torch.Tensor:
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
        out = x / nrm * math.sqrt(self.h.d)
        # (n_tokens, d)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See footnote under Equation 12 
        # a_comp = math.sqrt(1.0 - self.h.alpha_A**2)
        # m_comp = math.sqrt(1.0 - self.h.alpha_M**2)
        a_comp = 1.0
        m_comp = 1.0
        # Equation 4 (seen as Figure 1)
        x = a_comp*x + self.h.alpha_A*self.attn(self.layernorm(x))
        out = m_comp*x + self.h.alpha_M*self.mlp(self.layernorm(x))
        # (n_tokens, d)
        return out

@torch.no_grad()
def make_tokens(n: int, d: int, rho: float = 0.99, device="cpu") -> torch.Tensor:
    """
    E[||X_i||^2] = d, E[X_i*X_j] = rho*d, and E[p/q] = rho
    Appendix A
    """
    g  = torch.randn(d, device=device)
    Z  = torch.randn(n, d, device=device)
    X  = math.sqrt(rho) * g.unsqueeze(0) + math.sqrt(1.0 - rho) * Z
    # (n_tokens, d)
    return X

def get_qp(x: torch.Tensor) -> Tuple[float, float]:
    """ 
    Compute (q,p) for the expansion ratio
    """
    C = x @ x.T
    n = C.shape[0]
    diag = torch.diagonal(C)
    # diagonal average
    q = diag.mean().item()
    # off-diagonal average
    p = ((C.sum() - diag.sum()) / (n*(n-1))).item()
    return max(EPS, q), max(EPS, p)

@torch.no_grad()
def angle_lyapunov_exponent(h: Config, seeds: int = 1) -> float:
    """
    λ_a ≈ log((1 - p'/q') / (1 - p/q))
    Section 4.1
    """
    vals = []
    for s in range(seeds):
        torch.manual_seed(s)
        block = TransformerBlock(h).to(h.device).eval()
        x0 = make_tokens(h.n_tokens, h.d, device=h.device)
        q, p = get_qp(x0)
        r0 = max(EPS, 1.0 - p/q)
        x1 = block(x0)
        q1, p1 = get_qp(x1)
        r1 = max(EPS, 1.0 - p1/q1)
        vals.append(math.log(r1/r0))
    lambda_a = sum(vals)/len(vals)
    return lambda_a

def gradient_lyapunov_exponent(h: Config, depth: int = 16, seeds: int = 1) -> float:
    """
    λ_g ≈ (1/L) * log || d(X_L · R) / dX_0 ||_F^2
    Sections 3.4 & 4.2
    """
    vals = []
    for s in range(seeds):
        torch.manual_seed(s)
        blocks = nn.ModuleList([TransformerBlock(h).to(h.device).eval() for _ in range(depth)])
        X0 = make_tokens(h.n_tokens, h.d, rho=0.99, device=h.device).requires_grad_(True)
        Xt = X0
        for b in blocks:
            Xt = b(Xt)
        R = torch.randn_like(Xt)
        scalar = (Xt * R).sum()
        (dX,) = torch.autograd.grad(scalar, X0, retain_graph=False, create_graph=False, allow_unused=False)
        v = (dX**2).sum().item()
        vals.append(math.log(max(EPS, v)) / depth)
    lambda_g = sum(vals)/len(vals) 
    return lambda_g

if __name__ == "__main__":

    res = 50

    # edge of chaos covers where \lambda_a is close to 0
    # Figure 4 (top-right)
    alphas = torch.linspace(0, 1, res)
    sigmas = torch.linspace(1, 4, res)
    Z = torch.zeros(res, res)
    for i,sw in enumerate(sigmas):
        for j,a in enumerate(alphas):
            c = Config(alpha_A=a, alpha_M=a, sigma_w=sw)
            Z[i,j] = angle_lyapunov_exponent(c)

    vmin = float(Z.min().item())
    vmax = float(Z.max().item())
    cbar = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    extent = [float(alphas.min()), float(alphas.max()),
            float(sigmas.min()), float(sigmas.max())]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=130)
    plt.title('Numerical Token Angle Exponent')
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="bwr", norm=cbar, extent=extent)

    ax.set_xlabel(r"$\alpha = \alpha_A = \alpha_M$")
    ax.set_ylabel(r"$\sigma_w$")
    cb = fig.colorbar(im, ax=ax, label=r"$\lambda_a$")

    # draw λ_a=0 contour
    # CS = ax.contour(alphas, sigmas, Z, levels=[0.0], colors="k", linewidths=1.2)
    # ax.clabel(CS, fmt={0.0: r"$\lambda_a=0$"}, fontsize=8)

    plt.tight_layout()
    plt.show()