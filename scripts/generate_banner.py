from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import rosen, rosen_der

from dict_minimize.numpy_api import minimize


def rosen_obj(params):
    val = rosen(params["x"])

    dval = OrderedDict()
    dval["x"] = rosen_der(params["x"])
    return val, dval


np.random.seed(0)

banner = False
lr = 0.05

X = np.arange(-2.0, 2.0, 0.005)
Y = np.arange(-1.2, 1.2, 0.005)
X, Y = np.meshgrid(X, Y)

XY = np.stack((X, Y), axis=-1)
assert XY.shape[-1] == 2

Z = np.zeros_like(X)
for ii in range(X.shape[0]):
    for jj in range(X.shape[1]):
        Z[ii, jj] = rosen(XY[ii, jj, :])

params0 = OrderedDict()
params0["x"] = np.array([-0.1, -1.0])

log = []


def callback(xk):
    print(xk)
    print(rosen_der(xk["x"]))
    dr = rosen_der(xk["x"])
    log.append([xk["x"][0], xk["x"][1], -lr * dr[0], -lr * dr[1]])


params = minimize(rosen_obj, params0, method="L-BFGS-B", options={"disp": False}, callback=callback)

levels = np.percentile(Z.ravel(), np.linspace(5, 95, 20))

if banner:
    # For github README
    fig = plt.figure(figsize=(6, 2), dpi=320)
else:
    # for github social preview
    fig = plt.figure(figsize=(4, 2), dpi=320)

ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

CS = plt.contour(X, Y, Z, levels=levels, zorder=0)

for xx, yy, dx, dy in log:
    plt.arrow(xx, yy, dx, dy, zorder=1, color="silver")

xx, yy, dx, dy = zip(*log)
plt.plot(xx, yy, "r", zorder=2)
plt.plot(1, 1, "*", markersize=20, zorder=3)

plt.xticks(ticks=[])
plt.yticks(ticks=[])

plt.xlim(-1, 1.5)
plt.ylim(-0.5, 1.2)

plt.savefig("banner.png", dpi=320, pad_inches=0, bbox_inches="tight", transparent=False, format="png")
