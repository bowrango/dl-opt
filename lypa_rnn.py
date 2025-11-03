
# Basic implementation of the analysis presented in 
# "Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural Networks"

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import StudentT, Beta
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def init_weight_matrix(shape, dist="gaussian", nu=3.0, device="cpu"):
    if dist == "gaussian":
        W = torch.randn(*shape, device=device)
    elif dist == "studentT":
        t = StudentT(df=torch.tensor([nu], device=device))
        W = t.sample(shape).squeeze(-1)
        # Rescale to unit variance when nu>2 so "g" is comparable to Gaussian
        if nu > 2:
            W = W / math.sqrt(nu / (nu - 2.0))
        else:
            mad = torch.median(torch.abs(W - torch.median(W)))
            W = W / (mad + 1e-8) * 0.67449
    elif dist == "beta":
        # FIXME
        t = Beta(torch.FloatTensor([nu]), torch.FloatTensor([nu]))
        W = t.sample(shape).squeeze(-1)
    else:
        raise ValueError("unknown distribution")
    return W

def build_rnn(hidden_size=128, g=1.2, dist="gaussian", nu=3.0, device="cpu"):
    """
    Build a 1-layer tanh RNN with input_size=1 (we'll feed zeros or tiny noise).
    We set W_hh ~ (g / sqrt(n)) * iid entries (Gaussian or heavy-tailed),
    and W_ih small so input doesn't dominate unless you want it to.
    """
    n = hidden_size
    rnn = nn.RNN(input_size=1, hidden_size=n, nonlinearity="tanh", batch_first=False)
    rnn = rnn.to(device)
    with torch.no_grad():
        # Recurrent weights
        W_hh = init_weight_matrix((n, n), dist=dist, nu=nu, device=device)
        W_hh = (g / math.sqrt(n)) * W_hh
        rnn.weight_hh_l0.copy_(W_hh)

        # Input weights: small scale; set to zero if you want a purely autonomous system
        rnn.weight_ih_l0.normal_(mean=0.0, std=1e-3)
        # Biases
        if rnn.bias:
            rnn.bias_hh_l0.zero_()
            rnn.bias_ih_l0.zero_()
    return rnn

@torch.no_grad()
def lyapunov_spectrum_rnn(
    # quenched W
    rnn: nn.RNN,
    T=3000,
    warmup=500,
    h0=None,
    input_std=0.0,
    device="cpu",
    seed=0,
):
    """
    We propagate an orthonormal basis Q through J_t using QR and accumulate log|diag(R)|.
    """
    torch.manual_seed(seed)
    n = rnn.hidden_size
    W_hh = rnn.weight_hh_l0    # (n, n)
    if h0 is None:
        h = torch.randn(1, 1, n, device=device) * 0.1
    else:
        h = h0.reshape(1, 1, n).to(device)

    Q = torch.eye(n, device=device)
    gammas = torch.zeros(n, device=device)

    for t in range(T):
        # Small input drive (optional)
        if input_std > 0:
            x_t = torch.randn(1, 1, 1, device=device) * input_std
        else:
            x_t = torch.zeros(1, 1, 1, device=device)

        y, h = rnn(x_t, h)             # y: (1,1,n), h: (1,1,n)
        ht = y.squeeze(0).squeeze(0)   # (n,)
        if t >= warmup:
            # J_t = diag(1 - h_t^2) @ W_hh  (since nonlinearity is tanh)
            dphi = 1.0 - ht**2                       # (n,)
            J = (dphi.unsqueeze(1)) * W_hh           # row-scale W_hh by dphi
            Z = J @ Q
            Q, R = torch.linalg.qr(Z, mode="reduced")
            gammas += torch.log(torch.clamp(torch.abs(torch.diag(R)), min=1e-30))

    K = max(1, T - warmup)
    lambdas = gammas / K
    lambdas, _ = torch.sort(lambdas, descending=True)
    return lambdas.cpu().numpy()

@torch.no_grad()
def sweep_gain_for_top_exponent(
    n=128, gains=None, T=2000, warmup=400, dist="gaussian", nu=3.0, input_std=0.0, device="cpu", seed=0
):
    if gains is None:
        gains = np.linspace(0.6, 1.8, 13)
    tops = []
    for g in gains:
        rnn = build_rnn(hidden_size=n, g=g, dist=dist, nu=nu, device=device)
        lam = lyapunov_spectrum_rnn(rnn, T=T, warmup=warmup, input_std=input_std, device=device, seed=seed)
        tops.append(lam[0])
    return np.array(tops)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 128

    gains = np.linspace(0.6, 1.8, 13)

    # 1) Find edge of chaos for Gaussian weights
    top_gauss = sweep_gain_for_top_exponent(n=n, dist="gaussian", device=device, seed=1)
    idx = int(np.argmin(np.abs(top_gauss)))
    g_star = float(gains[idx])

    # 2) Full spectrum at g* (Gaussian)
    rnn_star = build_rnn(hidden_size=n, g=g_star, dist="gaussian", device=device)
    lam_star = lyapunov_spectrum_rnn(rnn_star, T=3000, warmup=500, device=device, seed=2)

    # 3) Heavy-tailed comparison (Student-t with ν=3)
    top_t = sweep_gain_for_top_exponent(n=n, dist="beta", nu=3.0, device=device, seed=1)
    idx_t = int(np.argmin(np.abs(top_t)))

    plt.plot(gains, top_gauss, label="Gaussian")
    plt.plot(gains, top_t, label="Student-t (ν=3)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("gain g")
    plt.ylabel("top Lyapunov exponent λ₁")
    plt.title("Edge of chaos")
    plt.legend()
    plt.show()