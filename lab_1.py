import math


def simple_iteration(g, f, x0, epsilon, max_iter=100):
    print("\nМетод простої ітерації")
    print(f"{'k':<5} {'x_k':<18} {'|x_k-x_{k-1}|':<18} {'|f(x_k)|':<18}")

    x_prev = x0
    x_next = 0.0

    for k in range(1, max_iter + 1):
        x_next = g(x_prev)
        error = abs(x_next - x_prev)
        f_val = abs(f(x_next))

        print(f"{k:<5} {x_next:<18.10f} {error:<18.10f} {f_val:<18.10f}")

        if error < epsilon:
            print(f"Збіжність до x = {x_next:.10f} за {k} ітерацій.")
            print(f"f(x) = {f(x_next):.10f}")
            return x_next

        x_prev = x_next

    print(f"Метод не зійшовся за {max_iter} ітерацій.")
    return x_prev


def newton_method(f, df, x0, epsilon, max_iter=100):
    print("\nМетод Ньютона")
    print(f"{'k':<5} {'x_k':<18} {'|x_k-x_{k-1}|':<18} {'|f(x_k)|':<18}")

    x_prev = x0
    x_next = 0.0

    for k in range(1, max_iter + 1):
        f_value = f(x_prev)
        df_value = df(x_prev)

        if abs(df_value) < 1e-12:
            print("Помилка: Похідна дорівнює нулю.")
            return None

        x_next = x_prev - f_value / df_value
        error = abs(x_next - x_prev)
        f_val_next = abs(f(x_next))

        print(f"{k:<5} {x_next:<18.10f} {error:<18.10f} {f_val_next:<18.10f}")

        if error < epsilon:
            print(f"Збіжність до x = {x_next:.10f} за {k} ітерацій.")
            print(f"f(x) = {f(x_next):.10f}")
            return x_next

        x_prev = x_next

    print(f"Метод не зійшовся за {max_iter} ітерацій.")
    return x_prev


# Головна частина програми
if __name__ == "__main__":
    print("Розв'язування: x + log10(1+x) - 1.5 = 0")

    epsilon = 0.001

    f = lambda x: x + math.log10(1 + x) - 1.5

    df = lambda x: 1 + 1 / ((x + 1) * math.log(10))

    lam = 0.911
    g = lambda x: x - lam * f(x)

    x0_iter = 1.0
    simple_iteration(g, f, x0_iter, epsilon)

    x0_newton = 1.0
    newton_method(f, df, x0_newton, epsilon)