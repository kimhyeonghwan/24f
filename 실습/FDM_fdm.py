#%%
import numpy as np 
import pandas as pd
from scipy.linalg import solve_banded, solve, solveh_banded
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

def exfdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds
    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    v = np.maximum(callOrPut*(s-k), 0)

    for j in range(M-1,-1,-1):
        temp = d * v[:-2] + (1 + m) * v[1:-1] + u * v[2:]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        v[1:-1] = temp
    f = interp1d(s,v)
    return pd.DataFrame({"S":s,"V":v}), f(s0)


#%%
s0, k, r, q, t, vol = 100, 100, 0.03, 0.01, 0.25, 0.4
optionType, maxS, N, M = "call", s0*2, 200, 2000
theta = 0

def fdm_vanilla_option(s0, k, r, q, t, vol, optionType, maxS, N, M, theta=1):
    ds = maxS / N
    dt = t / M
    callOrPut = 1 if optionType.lower()=='call' else -1

    i = np.arange(N+1)
    s = i * ds

    a = dt*(vol*s[1:-1])**2 / (2*ds**2)
    b = dt*(r-q)*s[1:-1] / (2*ds)
    d, m, u = a-b, -2*a-dt*r, a+b

    A = np.diag(d[1:],-1) + np.diag(m) + np.diag(u[:-1],1)
    B = np.zeros((N-1,2))
    B[0,0], B[-1,1] = d[0], u[-1] 

    Am = np.identity(N-1) - theta*A
    Ap = np.identity(N-1) + (1-theta)*A
    ab = np.zeros((3, N-1))
    ab[0,1:] = np.diag(Am,1)
    ab[1] = np.diag(Am)
    ab[2,:-1] = np.diag(Am,-1)

    v = np.maximum(callOrPut*(s-k), 0)
    for j in range(M-1,-1,-1):    
        # Too slow to calculate inverse matrix
        #temp = Ap @ v[1:-1] + theta*B @ v[[0,-1]]
        temp = (1-theta)*d * v[:-2] + (1 + (1-theta)*m) * v[1:-1] + (1-theta)*u * v[2:]
        temp[0] += theta*d[0]*v[0]
        temp[-1] += theta*u[-1]*v[-1]
        v[0] = np.maximum(callOrPut*(0 - k * np.exp(-r * (M - j) * dt)), 0)
        v[N] = np.maximum(callOrPut*(maxS - k * np.exp(-r * (M - j) * dt)), 0)
        temp += (1-theta)*B @ v[[0,-1]]
        # In tri-diagonal matrix, efficient solution "solve_banded"
        v[1:-1] = solve_banded((1,1), ab, temp)
        if j==1:
            f = interp1d(s,v)
            p1 = f(s0)

    f = interp1d(s,v)
    price = f(s0)

    return pd.DataFrame({"S":s,"V":v}), price

def fd_american_option(s0, k, r, q, t, sigma, option_type, n, m):
    # Set parameter
    maxS = s0 * 2  # max underlying price
    theta = 0.5  # Crank-Nicolson method
    omega = 1.2  # overrelaxation parameter
    threshold = 1e-8  # PSOR threshold

    ds = maxS / n
    dt = t / m 
    callOrPut = 1 if option_type.lower() == 'call' else -1

    i = np.arange(n + 1)
    s = i * ds

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
    theta = (f_next(s0)-f(s0)) / (dt * 250) # 1day theta

    return (price, delta, gamma, theta)

#%%
if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.00
    t = 0.25
    sigma = 0.3
    optionType = 'put'
    maxS, n, m = s*2, 100, 500
    v, ex_price = exfdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m)
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, ex_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0)
    print(f"EX-FDM Price = {ex_price:0.6f}")

    v, im_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m)
    print(f"IM-FDM Price = {im_price:0.6f}")

    v, cn_price = fdm_vanilla_option(s, k, r, q, t, sigma, optionType, 
                                    maxS, n, m, 0.5)
    print(f"CN-FDM Price = {cn_price:0.6f}")
    
    ame_price, ame_delta, ame_gamma, ame_theta = fd_american_option(s, k, r, q, t, sigma, optionType, 
                                    n, m)
    print(f"American-FDM Price = {ame_price:0.6f}")
    print(f"delta, gamma, theta = {ame_delta,ame_gamma,ame_theta}")

# %%
