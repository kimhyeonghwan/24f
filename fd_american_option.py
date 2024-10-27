# 20249132 김형환
# 금융수치해석기법 과제
# (1) PSOR 방법을 이용한 American plain vanilla option pricing
import numpy as np
from scipy.interpolate import interp1d

def fd_american_option(s0, k, r, q, t, sigma, option_type, n, m):
    # Set parameter
    maxS = s0 * 2  # max underlying price
    theta = 0.5  # Crank-Nicolson method
    omega = 1.2  # overrelaxation parameter
    threshold = 1e-8  # PSOR threshold

    ds = maxS / n
    dt = t / m 
    callOrPut = 1 if option_type.lower() == 'call' else -1

    s = np.arange(n + 1) * ds

    # FDM coefficients
    a = dt * (sigma * s[1:-1]) ** 2 / (2 * ds ** 2)
    b = dt * (r - q) * s[1:-1] / (2 * ds)
    d, m_, u = a - b, -2 * a - dt * r, a + b

    # implicit and explicit method (theta=0.5, CN method)
    A = np.diag(d[1:], -1) + np.diag(m_) + np.diag(u[:-1], 1)
    Am = np.identity(n - 1) - theta * A 
    Ap = np.identity(n - 1) + (1 - theta) * A

    ab = np.zeros((3, n - 1))
    ab[0, 1:] = np.diag(Am, 1)
    ab[1] = np.diag(Am)
    ab[2, :-1] = np.diag(Am, -1)

    v = np.maximum(callOrPut * (s - k), 0)

    for j in range(m - 1, -1, -1):
        v_next = np.copy(v) # For calculate greeks(theta)
        # Boundary condition in American options
        v[0] = np.maximum(-callOrPut * k, 0)  # S=0
        v[n] = np.maximum(callOrPut * (maxS - k * np.exp(-r * (m - j) * dt)), 0)  # S=maxS

        temp = (1 - theta) * d * v[:-2] + (1 + (1 - theta) * m_) * v[1:-1] + (1 - theta) * u * v[2:]
        temp[0] += theta * d[0] * v[0]
        temp[-1] += theta * u[-1] * v[-1]
        
        # PSOR method
        old_v = np.copy(v[1:-1])
        change = 2*threshold
        while change > threshold:
            change = 0
            for i in range(n - 1):
                residual = temp[i] - (Am[i, :] @ old_v)
                new_v = max(callOrPut * (s[i + 1] - k), old_v[i] + omega * residual / Am[i, i])
                current_change = abs(new_v - old_v[i])
                change = max(change, current_change)
                old_v[i] = new_v

        v[1:-1] = old_v

    f = interp1d(s, v)
    f_next = interp1d(s, v_next)
    price = f(s0)

    # Delta, Gamma, Theta
    delta = (f(s0 + ds) - f(s0 - ds)) / (2 * ds)
    gamma = (f(s0 + ds) - 2 * f(s0) + f(s0 - ds)) / (ds ** 2)
    theta = (f_next(s0) - f(s0)) / (dt * 250) # 1day theta

    return (price, delta, gamma, theta)