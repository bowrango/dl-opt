import numpy as np
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import glob
import os

# Use most recent file
files = glob.glob('sb_*.dat')
if not files:
    ValueError("No sb_*.dat files found")
latest_file = max(files, key=os.path.getmtime)

data = np.loadtxt(latest_file)
steps = data[:, 0]
min_energy = data[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(steps, min_energy, 'b-', linewidth=2, label='Min Energy')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Energy Evolution during Simulated Bifurcation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig(latest_file.replace('.dat', '.png'), dpi=150)
