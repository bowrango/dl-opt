import matplotlib.pyplot as plt
import numpy as np

d = 200
N = 100
sigma = 0.01

T = 2000
B = 16
eta = 0.01

np.random.seed(0)

X = np.random.normal(0, 1, (N,d))
ws = np.random.normal(0, 1, (d,1))
e = np.random.normal(0, sigma, (N,1))
y = X@ws + e

w = np.random.normal(0, 1, (d,1))

def L(w, X, y):
    return 0.5*np.mean((y - X@w)**2)

t = 0
err = np.empty(T)
while t < T:
    # perform an epoch with shuffling and no replacement
    perm = np.random.permutation(N)
    for idx in range(0, N, B):
        if t == T: break
        # compute batch loss
        batch = perm[idx:idx+B]
        xi, yi = X[batch], y[batch]
        loss = L(w, xi, yi)
        # compute batch gradient
        g = (xi.T@(xi@w - yi)) / len(xi)
        w -= eta*g
        err[t] = loss
        t += 1

plt.plot(err)
plt.title(f"SGD for Linear Regression (d={d},B={B})")
plt.xlabel('t')
plt.ylabel('loss')
plt.show()




# wb = 1
# sqrt3 = np.sqrt(3)

# def L(w):
#     return w*((w**2/3) - wb**2)
# def dL(w):
#     return w**2 - wb**2
# def H(x):
#     if np.abs(x) <= sqrt3:
#         return x
#     elif x > sqrt3:
#         return sqrt3
#     elif x < -sqrt3:
#         return -sqrt3
# def l2(x,y):
#     e = x-y
#     return np.dot(e,e)
    
# w1 = -0.9
# ws = -(2/3)*wb**3
# eta = 1/(2*sqrt3)

# eps = 1e-4
# G = 2*wb**2
# T = round((l2(w1,ws)*G**2) / eps)

# t = 0
# w = w1
# loss = np.inf
# W = [w1]
# LW = [L(w1)]
# while loss > eps:
#     if t == T: break
#     w = H(w - eta*dL(w))
#     Lw = L(w)
#     loss = l2(Lw, ws)
#     W.append(w)
#     LW.append(Lw)
#     t += 1

# x = np.linspace(-sqrt3*wb, sqrt3*wb, 1000)
# plt.plot(x, L(x))
# plt.scatter(W, LW, color="red", s=30, zorder=3)
# for step in range(t+1):
#     plt.text(W[step], LW[step], str(step), fontsize=8, ha="left", va="bottom")
# plt.legend()
# plt.xlabel("w")
# plt.ylabel("L(w)")
# plt.title(f"PGD after t={t}<={T} with eps={eps}")
# plt.show()