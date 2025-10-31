# https://kellerjordan.github.io/posts/muon/

import torch
from torch.optim import Optimizer

def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(Optimizer):
    """
    Momentum Orthogonalized by Newton-schulz
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # Update momentum buffer
                buf.mul_(momentum).add_(grad)

                # Apply orthogonalization for 2D parameters (matrices)
                if grad.ndim >= 2:
                    # Flatten to 2D if needed
                    original_shape = buf.shape
                    if buf.ndim > 2:
                        buf_2d = buf.view(buf.size(0), -1)
                    else:
                        buf_2d = buf

                    # Apply Newton-Schulz orthogonalization
                    buf_ortho = newtonschulz5(buf_2d, steps=ns_steps)

                    # Reshape back if needed
                    if buf.ndim > 2:
                        buf_ortho = buf_ortho.view(original_shape)

                    update = buf_ortho
                else:
                    # For 1D parameters (biases), just use momentum without orthogonalization
                    update = buf

                # Apply Nesterov momentum if enabled
                if nesterov:
                    update = grad + momentum * update

                # Update parameters
                p.data.add_(update, alpha=-lr)

        return loss