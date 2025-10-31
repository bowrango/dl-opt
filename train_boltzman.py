import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from models.boltzman import SimpleDBM

# Hello world script for training a Boltzmann network with p-bits [1]. 

# https://proceedings.mlr.press/v38/carlson15.pdf

class BarsAndStripes(Dataset):
    """
    Toy dataset with bars (vertical) and stripes (horizontal)
    """
    def __init__(self, size: int, num_samples: int):
        self.size = size
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.ones(self.size, self.size)
        if torch.rand(1) < 0.5:
            c = torch.randint(0, self.size, ())
            img[:, c] = -1
        else:
            r = torch.randint(0, self.size, ())
            img[r, :] = -1
        # Add channel dimension
        return img.unsqueeze(0)
    
class CrossesAndDiagonals(Dataset):
    """
    Toy dataset with crosses and diagonals
    """
    def __init__(self, size: int, num_samples: int):
        self.size = size
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.ones(self.size, self.size)
        choice = torch.randint(0, 3, ())
        if choice == 0:
            # Cross
            pt = torch.randint(0, self.size-1, ())
            img[pt, :] = -1
            img[:, pt] = -1
        elif choice == 1:
            # Main diagonal
            for i in range(self.size):
                img[i, i] = -1
        else:
            # Anti-diagonal
            for i in range(self.size):
                img[i, self.size - 1 - i] = -1
        # Add channel dimension
        return img.unsqueeze(0) 
    
class RandomBoxes(Dataset):
    """
    Toy dataset with randomly placed and sized binary rectangles (boxes).
    """
    def __init__(self, size: int, num_samples: int, min_box: int = 2, max_box: int = 6):
        self.size = size
        self.num_samples = num_samples
        self.min_box = min_box
        self.max_box = max_box
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.ones(self.size, self.size)
        # Random box size
        h = torch.randint(self.min_box, self.max_box + 1, ())
        w = torch.randint(self.min_box, self.max_box + 1, ())
        # Random top-left corner (ensure box fits)
        r = torch.randint(0, self.size - h + 1, ())
        c = torch.randint(0, self.size - w + 1, ())
        # Draw the box: set pixels inside the rectangle to -1
        img[r:r+h, c:c+w] = -1
        # Add channel dimension
        return img.unsqueeze(0)

if __name__ == "__main__":

    torch.manual_seed(0)

    sz = 16
    ds = BarsAndStripes(size=sz, num_samples=1000)
    # ds = CrossesAndDiagonals(size=sz, num_samples=1000)
    #ds = RandomBoxes(size=sz, num_samples=1000)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    n_h1 = 512
    n_h2 = 32
    net  = SimpleDBM(n_v=sz**2, n_h1=n_h1, n_h2=n_h2)

    # Train
    for epoch in range(1, 10):
        err = 0.0
        for images in loader:
            v = images.view(images.size(0), -1)
            loss = net(v, lr=0.01)
            #print(loss)
            err += loss.item()
        if epoch % 10 == 0:
            avg = err / len(loader)
            print(f"Epoch {epoch:04d}, mse={avg:.4f}")

    # Generate
    net.eval()
    num_gen = 5
    steps = 50
    fig, axes  = plt.subplots(1, num_gen, figsize=(num_gen*2,2))

    for i in range(num_gen):
        # Initialize pbits to random {-1,1}
        v  = 2*torch.bernoulli(torch.full((1,sz**2,), 0.5)) - 1
        h1 = 2*torch.bernoulli(torch.full((1, n_h1,), 0.5)) - 1
        h2 = 2*torch.bernoulli(torch.full((1, n_h2,), 0.5)) - 1

        for _ in range(steps):
            v, h1, h2 = net.gibbs(v, h1, h2)

        img = v.view(sz,sz).cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.show()