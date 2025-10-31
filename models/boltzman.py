import torch
import torch.nn as nn
import torch.nn.functional as F

# Boltzmann network is an energy-based model with visible and hidden nodes, and trained unsupervised using
# contrastive divergence to learn the data distribution.

# See more applications of p-bits [2].

# Some further work
# - beta tuning (APT)
# - solution quality degrades using LFSR vs sMTJ [3,4]

# Sources:
# [1] https://arxiv.org/pdf/2303.10728
# [2] https://arxiv.org/pdf/2302.06457
# [3] https://www.nature.com/articles/s41598-025-93218-8
# [4] https://arxiv.org/pdf/2505.19106
# [5] https://arxiv.org/pdf/2304.05949

class SimpleDBM(nn.Module):
    """
    Simple Deep Boltzmann Machine. It has 3-layers:
    (visible) - (hidden_1) - (hidden_2)
    The visible layer matches dimensions with the input data.
    """
    def __init__(self, n_v, n_h1, n_h2):
        super().__init__()
        self.W1  = nn.Parameter(0.01*torch.randn(n_h1, n_v))
        self.bv  = nn.Parameter(torch.zeros(n_v))
        self.bh1 = nn.Parameter(torch.zeros(n_h1))
        self.W2  = nn.Parameter(0.01*torch.randn(n_h2, n_h1))
        self.bh2 = nn.Parameter(torch.zeros(n_h2))

    def update(self, I, beta=1):
        # This function would be implemented with MAC and LUT
        # activations on the FPGA. The p-bit state is defined as
        # m = sign(tanh(beta*I) - U)
        # so the update probability is
        # P(m=+1) = P(tanh(beta*I) > U) = (tanh(beta*I)+1)/2 = sigmoid(2*beta*I)
        rng = torch.rand_like(I) # ~ N(0,1)
        bits = (torch.sigmoid(2*beta*I) > rng).float() # {0,1}
        pbits = 2*bits - 1 # {-1,1}
        return pbits

    def gibbs(self, v, h1, h2):
        # The update order follows the conditional layer dependencies.

        # First sample h1 ~ P(h1 | v, h2) using the bottom layer v and top layer h2
        l1 = F.linear(v, self.W1, self.bh1) + F.linear(h2, self.W2.t(), None)   
        h1 = self.update(l1)

        # Then sample h2 ~ P(h2 | h1) using only h1
        l2 = F.linear(h1, self.W2, self.bh2)
        h2 = self.update(l2)

        # Finally sample v  ~ P(v | h1) using only h1
        lv = F.linear(h1, self.W1.t(), self.bv)
        v  = self.update(lv)

        return v, h1, h2

    def forward(self, data, lr=0.01):
        # Positive data phase clamps visible pbits to the data
        h1p = self.update(F.linear(data, self.W1, self.bh1))
        h2p = self.update(F.linear(h1p, self.W2, self.bh2))

        # Negative model phase runs without clamping
        vn, h1n, h2n = self.gibbs(data, h1p, h2p)

        # Update using Eqs. (5-6) [2]
        batch = data.size(0)
        self.W1.data += lr*((h1p.t() @ data - h1n.t() @ vn) / batch)
        self.W2.data += lr*((h2p.t() @ h1p - h2n.t() @ h1n) / batch)
        self.bv.data += lr*(data - vn).mean(0)
        self.bh1.data += lr*(h1p - h1n).mean(0)
        self.bh2.data += lr*(h2p - h2n).mean(0)

        return F.mse_loss(vn, data)