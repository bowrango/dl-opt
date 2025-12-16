# EF-SGD vs Naive Compressed-SGD vs plain SGD (linear regression)
import torch, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(7); np.random.seed(7)

# data (easy ground-truth)
n, d = 1500, 80
X = torch.randn(n, d)
true_w = torch.randn(d)
y = X @ true_w + 0.1 * torch.randn(n)

def loss_and_grad(w, idx):
    xb, yb = X[idx], y[idx]
    resid = xb @ w - yb
    return 0.5 * (resid**2).mean(), (xb.T @ resid) / xb.shape[0]

# Lipschitz L of grad_f for squared loss via power iteration
with torch.no_grad():
    A = X / math.sqrt(n)
    v = torch.randn(d)
    for _ in range(30):
        v = A.T @ (A @ v); v = v / (v.norm() + 1e-12)
    L = (v @ (A.T @ (A @ v))).item()

# compressor: biased top-k sparsifier
def topk_compress(vec, k):
    if k >= vec.numel(): return vec.clone(), 1.0
    vals = torch.topk(vec.abs(), int(k), sorted=False).values
    thresh = vals.min()
    mask = (vec.abs() >= thresh)
    out = torch.where(mask, vec, torch.zeros_like(vec))
    return out, mask.float().mean().item()


T, batch, eta = 200, 64, 0.06
k_keep = max(1, d // 20)
perm = torch.randint(0, n, (T, batch))

w_sgd = torch.zeros(d)
w_c   = torch.zeros(d)
w_ef  = torch.zeros(d)
e_buf = torch.zeros(d)   # EF residual

log = {k: [] for k in
       ["t","loss_sgd","loss_c","loss_ef",
        "nnz_c","nnz_ef","cerr_c","cerr_ef",
        "bias_c","bias_ef","var_c","var_ef"]}

# online bias/variance trackers for compression noise
running_mean_c = torch.zeros(d)
running_mean_ef = torch.zeros(d)
running_sq_c = torch.zeros(d)
running_sq_ef = torch.zeros(d)
count = 0

for t in range(T):
    idx = perm[t]
    _, g_sgd = loss_and_grad(w_sgd, idx)
    _, g_c = loss_and_grad(w_c,   idx)
    _, g_efg = loss_and_grad(w_ef,  idx)

    # 1) plain SGD
    w_sgd = w_sgd - eta * g_sgd
    loss_sgd, _ = loss_and_grad(w_sgd, idx)

    # 2) naive compressed-SGD
    cvec_c, r_c = topk_compress(g_c, k_keep)
    w_c = w_c - eta * cvec_c
    loss_c, _ = loss_and_grad(w_c, idx)

    # 3) EF-SGD (error feedback)
    g_send = g_efg + e_buf
    cvec_ef, r_ef = topk_compress(g_send, k_keep)
    e_buf = g_send - cvec_ef              # store the error
    w_ef = w_ef - eta * cvec_ef
    loss_ef, _ = loss_and_grad(w_ef, idx)

    # diagnostics
    diff_c  = cvec_c  - g_c
    diff_ef = cvec_ef - g_efg
    count += 1
    running_mean_c  += (diff_c  - running_mean_c ) / count
    running_mean_ef += (diff_ef - running_mean_ef) / count
    running_sq_c    += (diff_c  ** 2 - running_sq_c ) / count
    running_sq_ef   += (diff_ef ** 2 - running_sq_ef) / count
    var_c  = (running_sq_c  - running_mean_c  ** 2).mean().item()
    var_ef = (running_sq_ef - running_mean_ef ** 2).mean().item()

    log["t"].append(t)
    log["loss_sgd"].append(loss_sgd.item())
    log["loss_c"].append(loss_c.item())
    log["loss_ef"].append(loss_ef.item())
    log["nnz_c"].append(r_c);           log["nnz_ef"].append(r_ef)
    log["cerr_c"].append(diff_c.norm().item())
    log["cerr_ef"].append(diff_ef.norm().item())
    log["bias_c"].append(running_mean_c.norm().item())
    log["bias_ef"].append(running_mean_ef.norm().item())
    log["var_c"].append(var_c);         log["var_ef"].append(var_ef)

import pandas as pd
df = pd.DataFrame(log)
print(df.head(10)[["t","loss_sgd","loss_c","loss_ef","nnz_c","nnz_ef"]])

plt.figure(figsize=(8,4))
plt.plot(df["loss_sgd"], label="SGD (no comp)")
plt.plot(df["loss_c"],   label="Naive Compressed-SGD")
plt.plot(df["loss_ef"],  label="EF-SGD")
plt.xlabel("iteration"); plt.ylabel("mini-batch loss"); plt.title("Training loss")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["cerr_c"],  label="‖C(g)-g‖ (Naive-C)")
plt.plot(df["cerr_ef"], label="‖C(g+e)-g‖ (EF-SGD)")
plt.xlabel("iteration"); plt.ylabel("compression error norm")
plt.title("Per-step information loss"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["bias_c"],  label="bias proxy (Naive-C)")
plt.plot(df["bias_ef"], label="bias proxy (EF-SGD)")
plt.xlabel("iteration"); plt.ylabel("‖E[C(·)]-g‖ (running proxy)")
plt.title("Bias of compressed signal vs true gradient")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["var_c"],  label="variance proxy (Naive-C)")
plt.plot(df["var_ef"], label="variance proxy (EF-SGD)")
plt.xlabel("iteration"); plt.ylabel("per-coordinate variance")
plt.title("Variance of compression noise")
plt.legend(); plt.tight_layout(); plt.show()
