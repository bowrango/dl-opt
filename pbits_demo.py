import torch
import matplotlib.pyplot as plt

# Basic demo of pbit equations showing convergence to Boltzmann distribution

# Sources:
# [1] https://arxiv.org/pdf/2302.06457
# [2] https://arxiv.org/pdf/2303.10728

numBits = 8
numSteps = int(1e3)
# betas = torch.tensor([1e-2, 4e-2, 16e-2, 64e-2])
betas = torch.tensor([1e-3, 1e-2, 1e-1, 1])
Q = 1

torch.manual_seed(0)

def stateindex(pbits):
    bits = ((pbits + 1) / 2).to(torch.int32)
    return int(''.join(str(b.item()) for b in bits), 2)

def boltzmann(W, h, beta):
    numBits = W.size(0)
    states = torch.cartesian_prod(*([torch.tensor([-1., 1.])] * numBits))
    E = (states @ W * states).sum(dim=1) + (states * h).sum(dim=1)
    return torch.softmax(-beta * E, dim=0)

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
axs = axs.flatten()

for ii, beta in enumerate(betas):
    all_probs = []
    all_boltz = []

    for q in range(Q):
        W = torch.randn(numBits, numBits)
        W = W + W.T
        W.fill_diagonal_(0)
        h = torch.randn(numBits)

        pbits = torch.sign(torch.randn(numBits))
        counts = torch.zeros(2**numBits)

        for _ in range(numSteps):
            for jj in range(numBits):
                I = W[jj] @ pbits + h[jj]
                rng = 2 * torch.rand(1) - 1
                pbits[jj] = torch.sign(torch.tanh(beta * I) - rng)
            counts[stateindex(pbits)] += 1
        probs /= counts.sum()
        all_probs.append(probs)

        B = boltzmann(W, h, beta)
        all_boltz.append(B)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    avg_boltz = torch.stack(all_boltz).mean(dim=0)

    for p in all_probs:
        axs[ii].plot(torch.sort(p).values.numpy(), 'r.', alpha=0.3)
    for b in all_boltz:
        axs[ii].plot(torch.sort(b).values.numpy(), 'k-' , alpha=0.3)

    axs[ii].set_title(rf'$\beta = {beta:.3f}$')
    axs[ii].set_xlabel('State')
    axs[ii].set_ylabel('Probability')
    axs[ii].legend()
    axs[ii].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()