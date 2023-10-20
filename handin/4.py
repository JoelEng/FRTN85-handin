import math
import numpy as np
from itertools import islice
from matplotlib import pyplot as plt 

a_12 = 2 # a_1 and a_2 are the same
points = [("A", 1, 3), ("B", 2, 1), ("C", 1, 1), ("D", 3, 2)]

def theta_2(x, y):
    dividend = (x ** 2 + y ** 2)
    quotient = (8 ** 2) - dividend
    return 2 * math.atan(math.sqrt(quotient / dividend))

def phi(x, y):
    return math.atan2(y, x)

def psi(x, y):
    theta = theta_2(x, y)
    return math.atan2(2 * math.sin(theta), 2 + 2 * math.cos(theta))

def theta_1(x, y):
    return phi(x, y) - psi(x, y)

def theta_vals():
    t1_vals = [0]
    t2_vals = [0]
    for (name, x, y) in points:
        t1 = theta_1(x, y)
        t2 = theta_2(x, y)
        t1_vals.append(t1)
        t2_vals.append(t2)
        print(name)
        print("  θ1 =", t1)
        print("  θ2 =", t2)
    return (t1_vals, t2_vals)

(t1_vals, t2_vals) = theta_vals()
print("\n")

def poly_coefs(q0, qf):
    t0 = 0
    tf = 1
    left = np.array([
        [1, t0, t0 ** 2, t0 ** 3],
        [0, 1, 2 * t0, 3 * t0 ** 2],
        [1, tf, tf ** 2, tf ** 3],
        [0, 1, 2 * tf, 3 * tf ** 2]
    ])
    right = np.array([[q0], [0], [qf], [0]])
    a_vals = np.linalg.solve(left, right)
    a_flat = [x for sub in a_vals for x in sub]
    return a_flat

time = np.arange(0, 1, 0.01)

def poly_trajectory(q0, qf):
    a = poly_coefs(q0, qf)
    ar = [round(x, 2) for x in a]
    print(f"  q(t) = {ar[0]} + {ar[1]}t + {ar[2]}t² + {ar[3]}t³")
    print(f"  q'(t) = {ar[1]} + {2 * ar[2]}t + {3 * ar[3]}t²")
    return (
        a[0] + a[1] * time + a[2] * time ** 2 + a[3] * time ** 3,
        a[1] + 2 * a[2] * time + 3 * a[3] * time ** 2
    )

def plot(plot_name, qt, qt_p):
    plt.title(plot_name)
    plt.xlabel("time")
    plt.ylabel("angular position")
    plt.plot(time, qt)
    plt.savefig(f"4cplots/{plot_name}")
    # plt.show()
    
for name, q01, qf1, q02, qf2 in zip(["A", "B", "C", "D"], t1_vals, t1_vals[1:], t2_vals, t2_vals[1:]):
    print(name, "θ1")
    (qt, qt_p) = poly_trajectory(q01, qf1)
    plot(f"{name}θ1", qt, qt_p)

    print(name, "θ2")
    (qt, qt_p) = poly_trajectory(q02, qf2)
    plot(f"{name}θ2", qt, qt_p)
