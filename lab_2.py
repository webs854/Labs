import numpy as np

A = np.array([
    [-0.87,  0.22, -0.33,  0.07],
    [ 0.00, -1.55, -0.23,  0.07],
    [ 0.11,  0.00, -1.08,  0.78],
    [ 0.08,  0.09,  0.33, -0.79]
])

B = np.array([-0.11, 0.33, -0.85, 1.70])

eps = 1e-4

def gauss_seidel(A, B, eps):
    n = len(B)
    x = np.zeros(n)
    x_new = np.zeros(n)

    print("Метод Зейделя")
    print(f"{'k':<4} {'x1':>12} {'x2':>12} {'x3':>12} {'x4':>12} {'|ΔX|':>12}")

    k = 0
    while True:
        k += 1
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (B[i] - s1 - s2) / A[i][i]

        diff = np.max(np.abs(x_new - x))

        print(f"{k:<4} {x_new[0]:>12.8f} {x_new[1]:>12.8f} {x_new[2]:>12.8f} {x_new[3]:>12.8f} {diff:>12.8f}")

        if diff < eps:
            break

        x = x_new.copy()

    return x_new, k

def simple_iteration(A, B, eps):
    n = len(B)
    C = np.zeros_like(A, dtype=float)
    D = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] / A[i][i]
        D[i] = B[i] / A[i][i]

    x = np.zeros(n)
    x_new = np.zeros(n)
    print("Метод простої ітерації")
    print(f"{'k':<4} {'x1':>12} {'x2':>12} {'x3':>12} {'x4':>12} {'|ΔX|':>12}")

    k = 0
    while True:
        k += 1

        x_new = D + C.dot(x)
        diff = np.max(np.abs(x_new - x))

        print(f"{k:<4} {x_new[0]:>12.8f} {x_new[1]:>12.8f} {x_new[2]:>12.8f} {x_new[3]:>12.8f} {diff:>12.8f}")

        if diff < eps:
            break

        x = x_new.copy()

    return x_new, k


x_gs, it_gs = gauss_seidel(A, B, eps)
x_si, it_si = simple_iteration(A, B, eps)

print("Підсумки")
print(f"Метод Зейделя:         X = {x_gs},    ітерацій = {it_gs}")
print(f"Метод простої ітерації X = {x_si},    ітерацій = {it_si}")
