
# Empirical proof for SGD with debiased momentum

# The inequality (in expectation over z_t | w_t) is approximated empirically:
#   ||e_{t+1}||^2  <=  (1 - beta)^2 ||e_t||^2  +  2 beta^2 sigma^2  +  2 (1 - beta)^2 L^2 eta^2 ||m_t||^2
#
# where:
#   e_t = m_t - grad f(w_t)
#   sigma^2 = E_z || grad l(w_t, z) - grad f(w_t) ||^2  (we compute it exactly over the dataset)
#   L is the Lipschitz constant of grad f; for squared loss it's the largest eigenvalue of (X^T X)/n

# decay of past error (1-\beta)^2
# stochastic variance 2\beta^2\sigma^2
# drift 2(1-\beta)^2L^2\eta^2\|m_t\|^2

import torch
import math
import matplotlib.pyplot as plt

torch.manual_seed(0)

n = 1000
d = 5
X = torch.randn(n, d)
true_w = torch.randn(d)
y = X @ true_w + 0.1 * torch.randn(n)

# Loss for a single sample z=(x_i,y_i):
#   l(w, z) = 0.5 * (x^T w - y)^2
# grad l(w, z) = x * (x^T w - y)
def grad_l_sample(w, x, y):
    return x * (x @ w - y)

# Population objective:
#   f(w) = (1/(2n)) sum_i (x_i^T w - y_i)^2
# grad f(w) = (1/n) X^T (X w - y)
def grad_f(w):
    return (X.T @ (X @ w - y)) / X.shape[0]

# Lipschitz constant L of grad f for squared loss is max eigenvalue of (X^T X)/n
# which equals (max singular value of X / sqrt(n))^2
with torch.no_grad():
    # Compute spectral norm of X / sqrt(n) with power iteration
    A = X / math.sqrt(n)
    v = torch.randn(d)
    for _ in range(50):
        v = (A.T @ (A @ v))
        v = v / (v.norm() + 1e-12)
    Lv = v
    L_est = (Lv @ (A.T @ (A @ Lv))).item()  # Rayleigh quotient ~ max eigenvalue
L = L_est

# Hyperparameters
beta = 0.8
eta = 0.02
T = 200  # number of steps

# Initialize
w = torch.zeros(d)
m = torch.zeros(d)
e_hist = []
lhs_hist = []
rhs_hist = []
sigma_hist = []
mt_norm_hist = []

# Precompute helpers to get sigma^2(w) exactly over the dataset
def sigma2_at_w(w):
    gbar = grad_f(w)  # mean gradient
    diffs = X * ((X @ w - y).unsqueeze(1)) - gbar  # each row: grad_l_i(w) - grad_f(w)
    # squared norm per sample, then mean:
    sq = (diffs ** 2).sum(dim=1)
    return sq.mean().item()

# Simulate debiased momentum
for t in range(T):
    # Sample current z_t uniformly
    idx = torch.randint(0, n, (1,)).item()
    x_t = X[idx]
    y_t = y[idx]

    # Compute grad at w_t and correction at w_{t-1} using same (x_t, y_t)
    g_wt = grad_l_sample(w, x_t, y_t)
    # For the correction term, we need grad l(w_{t-1}, z_t).
    # We'll keep previous w and m cached from last step. For t=0, use zeros (consistent).
    if t == 0:
        w_prev = w.clone()
        g_wprev = grad_l_sample(w_prev, x_t, y_t)
    else:
        g_wprev = grad_l_sample(w_prev, x_t, y_t)

    # Debiased momentum update
    m_new = g_wt + beta * (m - g_wprev)

    # Track approximation error e_t and quantities for inequality
    e_t = m - grad_f(w)  # e_t defined at *current* w_t and m_t
    e_hist.append(e_t.norm().item())

    # Compute sigma^2 at w_t (exact over dataset)
    sigma2 = sigma2_at_w(w)
    sigma_hist.append(sigma2)

    # Compute inequality RHS terms using current e_t, m_t, sigma^2(w_t)
    # Note: the theoretical inequality is in conditional expectation over z_t.
    # Empirically we compare the realized LHS with RHS that uses exact sigma^2(w_t) and L.
    rhs = (1 - beta) ** 2 * e_t.norm().pow(2).item() \
          + 2 * (beta ** 2) * sigma2 \
          + 2 * ((1 - beta) ** 2) * (L ** 2) * (eta ** 2) * (m.norm().pow(2).item())
    rhs_hist.append(rhs)
    mt_norm_hist.append(m.norm().item())

    # Compute next error and LHS after forming m_{t+1}
    e_next = m_new - grad_f(w)  # e_{t+1} but using grad_f(w_t) (matches the conditional structure)
    lhs_hist.append(e_next.norm().pow(2).item())

    # Update parameters for next step
    w_prev = w.clone()
    w = w - eta * m  # use current m_t to update w_{t+1}
    m = m_new

print(f"Lipschitz L (estimated): {L:.4f}")

# Plot LHS and RHS over iterations
plt.figure(figsize=(8,4))
plt.plot(lhs_hist, label="LHS  = ||e_{t+1}||^2")
plt.plot(rhs_hist, label="RHS bound")
plt.xlabel("iteration t")
plt.legend()
plt.tight_layout()
plt.show()