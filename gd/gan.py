import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

q_mean = torch.tensor([3.0, 4.0])
d = 2

approx = False
if approx:
    N = 100
    z = torch.randn(N, d) + q_mean.unsqueeze(0)
    Z = torch.randn(N, d)
    mean_z = z.mean(dim=0)
    mean_Z = Z.mean(dim=0)
    mean_diff = mean_z - mean_Z
else:
    # (E[z] - E[Z] = q_mean - 0)
    mean_diff = q_mean

# min x -> gradient descent
# max y -> gradient ascent

def U(x, y):
    return torch.dot(y, mean_diff - x)

def grad_x(x, y):
    return -y.clone()

def grad_y(x, y):
    return mean_diff - x

# projection to enforce Lipschitz ||y|| <= 1
def proj_y(y, radius=1.0):
    norm = y.norm()
    if norm > radius:
        return y * (radius / norm)
    return y

x0 = torch.tensor([1.0, 1.0])
y0 = torch.tensor([1.0, 1.0])
T = 100

x_star = q_mean.clone().numpy()
y_star = np.zeros(d)

# 1) Vanilla Gradient (GD)
lr_x_gd = 0.07
lr_y_gd = 0.05

x_gd = x0.clone()
y_gd = y0.clone()
x_hist_gd = [x_gd.numpy()]
y_hist_gd = [y_gd.numpy()]

for t in range(T):
    gx = grad_x(x_gd, y_gd)
    gy = grad_y(x_gd, y_gd)

    x_gd = x_gd - lr_x_gd * gx
    y_gd = y_gd + lr_y_gd * gy
    y_gd = proj_y(y_gd)

    x_hist_gd.append(x_gd.numpy())
    y_hist_gd.append(y_gd.numpy())

# 2) Proximal Point (PP)
lr_x_prox = 0.07
lr_y_prox = 0.05

x_prox = x0.clone()
y_prox = y0.clone()
x_hist_prox = [x_prox.numpy()]
y_hist_prox = [y_prox.numpy()]

for t in range(T):
    gx = grad_x(x_prox, y_prox)
    gy = grad_y(x_prox, y_prox)

    x_prox = x_prox - lr_x_prox * gx
    y_prox = y_prox + lr_y_prox * gy
    y_prox = proj_y(y_prox)

    x_hist_prox.append(x_prox.numpy())
    y_hist_prox.append(y_prox.numpy())

# 3) Extragradient (EG)
lr_x_eg = 0.1
lr_y_eg = 0.1

x_eg = x0.clone()
y_eg = y0.clone()
x_hist_eg = [x_eg.numpy()]
y_hist_eg = [y_eg.numpy()]

for t in range(T):
    gx = grad_x(x_eg, y_eg)
    gy = grad_y(x_eg, y_eg)

    x_mid = x_eg - lr_x_eg * gx
    y_mid = y_eg + lr_y_eg * gy

    gx_mid = grad_x(x_mid, y_mid)
    gy_mid = grad_y(x_mid, y_mid)

    x_eg = x_eg - lr_x_eg * gx_mid
    y_eg = y_eg + lr_y_eg * gy_mid
    y_eg = proj_y(y_eg)

    x_hist_eg.append(x_eg.numpy())
    y_hist_eg.append(y_eg.numpy())

# 4) Optimistic Gradient Descent (OGD)
lr_x_ogd = 0.1
lr_y_ogd = 0.1

x_ogd = x0.clone()
y_ogd = y0.clone()
gx_prev = torch.zeros_like(x_ogd)
gy_prev = torch.zeros_like(y_ogd)

x_hist_ogd = [x_ogd.numpy()]
y_hist_ogd = [y_ogd.numpy()]

for t in range(T):
    gx_curr = grad_x(x_ogd, y_ogd)
    gy_curr = grad_y(x_ogd, y_ogd)

    x_ogd = x_ogd - 2.0*lr_x_ogd*gx_curr + lr_x_ogd*gx_prev
    y_ogd = y_ogd + 2.0*lr_y_ogd*gy_curr - lr_x_ogd*gy_prev

    y_ogd = proj_y(y_ogd)

    gx_prev = gx_curr
    gy_prev = gy_curr

    x_hist_ogd.append(x_ogd.numpy())
    y_hist_ogd.append(y_ogd.numpy())

# Visualization
methods = [
    ('GD', np.array(x_hist_gd), np.array(y_hist_gd)),
    ('PP', np.array(x_hist_prox), np.array(y_hist_prox)),
    ('EG', np.array(x_hist_eg), np.array(y_hist_eg)),
    ('OGD', np.array(x_hist_ogd), np.array(y_hist_ogd)),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, (name, x_hist, y_hist) in enumerate(methods):
    ax = axes[idx // 2, idx % 2]

    ax.plot(x_hist[:, 0], x_hist[:, 1], 'b-o', markersize=3, alpha=0.6, label='x trajectory')
    ax.plot(y_hist[:, 0], y_hist[:, 1], 'r-s', markersize=3, alpha=0.6, label='y trajectory')

    ax.plot(x_hist[0, 0], x_hist[0, 1], 'bo', markersize=8, label='x init')
    ax.plot(x_hist[-1, 0], x_hist[-1, 1], 'bX', markersize=12, label='x final')

    ax.plot(y_hist[0, 0], y_hist[0, 1], 'ro', markersize=8, label='y init')
    ax.plot(y_hist[-1, 0], y_hist[-1, 1], 'rX', markersize=12, label='y final')

    ax.plot(x_star[0], x_star[1], 'b*', markersize=12, label='x*')
    ax.plot(y_star[0], y_star[1], 'r*', markersize=12, label='y*')

    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, x_hist, y_hist in methods:
    dist_x = np.linalg.norm(x_hist - x_star.reshape(1, -1), axis=1)
    ax1.plot(dist_x, label=name, linewidth=2)
ax1.set_yscale('log')
ax1.set_title('||x - x*||')
ax1.legend()
ax1.grid(True, alpha=0.3)

for name, x_hist, y_hist in methods:
    dist_y = np.linalg.norm(y_hist - y_star.reshape(1, -1), axis=1)
    ax2.plot(dist_y, label=name, linewidth=2)
ax2.set_yscale('log')
ax2.set_title('||y - y*||')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()