# 20249132 김형환
# 금융수치해석기법 과제
# (3) 2-factor FDM의 ADI 및 OSM 방식으로 Binary options 평가

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import RectBivariateSpline

def adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    # Set parameter
    smax1, smax2 = s1 * 2, s2 * 2 # max underlying price
    ds1, ds2, dt = smax1 / nx, smax2 / ny, t / nt
    s1_v, s2_v = np.arange(nx + 1) * ds1, np.arange(ny + 1) * ds2
    s_1, s_2 = np.meshgrid(s1_v, s2_v)

    # Set payoff : 10000
    v = np.zeros((ny + 1, nx + 1))
    v = np.where(np.minimum(s_1, s_2) >= k, 10000, v)
    v = np.where((np.minimum(s_1, s_2) >= k - 1 / oh) & (np.minimum(s_1, s_2) < k),
                 10000 * oh * (np.minimum(s_1, s_2) - (k - 1 / oh)), v)

    # FDM coefficients
    a1 = (dt * sigma1**2 * s_1**2) / (4 * ds1**2)
    b1 = (dt * (r - q1) * s_1) / (4 * ds1)
    d1 = - (a1[:, 1:-1] + b1[:, 1:-1])
    u1 = - (a1[:, 1:-1] - b1[:, 1:-1])
    m1 = 1 + 2 * a1[:, 1:-1] + dt * r / 2
    a2 = (dt * sigma2**2 * s_2**2) / (4 * ds2**2)
    b2 = (dt * (r - q2) * s_2) / (4 * ds2)
    d2 = - (a2[1:-1, :] + b2[1:-1, :])
    u2 = - (a2[1:-1, :] - b2[1:-1, :])
    m2 = 1 + 2 * a2[1:-1, :] + dt * r / 2
    c = dt * corr * sigma1 * sigma2 * s_1 * s_2 / (8 * ds1 * ds2)

    for n in range(nt - 1, -1, -1):
        v_next = v
        # implicit in x, explicit in y
        for j in range(1, ny):
            ex_y_ij = v[j, 1:-1] + c[j, 1:-1] * (v[j+1, 2:] - v[j-1, 2:] - v[j+1, :-2] + v[j-1, :-2]) + \
                   a2[j, 1:-1] * (v[j+1, 1:-1] - 2 * v[j, 1:-1] + v[j-1, 1:-1]) + \
                   b2[j, 1:-1] * (v[j+1, 1:-1] - v[j-1, 1:-1])
            ex_y_ij[0] -= u1[j, 0] * v[j, 0]
            ex_y_ij[-1] -= d1[j, -1] * v[j, -1]
            # Generate tri-diagonal matrix
            ab_x = np.zeros((3, nx - 1))
            ab_x[0, 1:] = d1[j, :-1]
            ab_x[1] = m1[j, :]
            ab_x[2, :-1] = u1[j, 1:]
            v[j, 1:-1] = solve_banded((1, 1), ab_x, ex_y_ij)

        # implicit in y, explicit in x
        for i in range(1, nx):
            ex_x_ij = v[1:-1, i] + c[1:-1, i] * (v[2:, i+1] - v[:-2, i+1] - v[2:, i-1] + v[:-2, i - 1]) + \
                   a1[1:-1, i] * (v[1:-1, i+1] - 2 * v[1:-1, i] + v[1:-1, i-1]) + \
                   b1[1:-1, i] * (v[1:-1, i+1] - v[1:-1, i-1])
            ex_x_ij[0] -= u2[0, i] * v[0, i]
            ex_x_ij[-1] -= d2[-1, i] * v[-1, i]
            # Generate tri-diagonal matrix
            ab_y = np.zeros((3, ny - 1))
            ab_y[0, 1:] = d2[:-1, i]
            ab_y[1] = m2[:, i]
            ab_y[2, :-1] = u2[1:, i]
            v[1:-1, i] = solve_banded((1, 1), ab_y, ex_x_ij)

        # Boundary condition
        v[0, :] = 2 * v[1, :] - v[2, :]
        v[-1, :] = 2 * v[-2, :] - v[-3, :]
        v[:, 0] = 2 * v[:, 1] - v[:, 2]
        v[:, -1] = 2 * v[:, -2] - v[:, -3]
        v[0, 0] = v[0, 1] + v[1, 0] - v[1, 1]
        v[-1, 0] = v[-2, 0] + v[-1, 1] - v[-2, 1]
        v[0, -1] = v[0, -2] + v[1, -1] - v[1, -2]
        v[-1, -1] = v[-1, -2] + v[-2, -1] - v[-2, -2]

    f = RectBivariateSpline(s2_v, s1_v, v)
    f_next = RectBivariateSpline(s2_v, s1_v, v_next)
    
    # Greeks
    price = f(s2, s1)[0, 0]
    delta1 = (f(s2, s1 + ds1)[0, 0] - f(s2, s1 - ds1)[0, 0]) / (2 * ds1)
    delta2 = (f(s2 + ds2, s1)[0, 0] - f(s2 - ds2, s1)[0, 0]) / (2 * ds2)
    gamma1 = (f(s2, s1 + ds1)[0, 0] - 2 * price + f(s2, s1 - ds1)[0, 0]) / (ds1**2)
    gamma2 = (f(s2 + ds2, s1)[0, 0] - 2 * price + f(s2 - ds2, s1)[0, 0]) / (ds2**2)
    cross_gamma = (f(s2 + ds2, s1 + ds1)[0, 0] - f(s2 + ds2, s1 - ds1)[0, 0] -
                   f(s2 - ds2, s1 + ds1)[0, 0] + f(s2 - ds2, s1 - ds1)[0, 0]) / (4 * ds1 * ds2)
    theta = (f_next(s2, s1)[0, 0] - f(s2, s1)[0, 0]) / (dt * 250) # 1day theta

    return (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)

def osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    # Set parameter
    smax1, smax2 = s1*2, s2*2 # max underlying price
    ds1, ds2, dt = smax1 / nx, smax2 / ny, t / nt
    s1_v, s2_v = np.arange(nx + 1) * ds1, np.arange(ny + 1) * ds2
    s_1, s_2 = np.meshgrid(s1_v, s2_v)

    # Set payoff : 10000
    v = np.zeros((ny + 1, nx + 1))
    v = np.where(np.minimum(s_1, s_2) >= k, 10000, v)
    v = np.where((np.minimum(s_1, s_2) >= k - 1 / oh) & (np.minimum(s_1, s_2) < k),
                 10000 * oh * (np.minimum(s_1, s_2) - (k - 1 / oh)), v)

    # FDM coefficients
    a1 = (dt * sigma1**2 * s_1**2) / (2 * ds1**2)
    b1 = (dt * (r - q1) * s_1) / (2 * ds1)
    d1 = - (a1[:,1:-1] + b1[:,1:-1])
    u1 = - (a1[:,1:-1] - b1[:,1:-1])
    m1 = 1 + 2*a1[:,1:-1] + dt * r / 2
    a2 = (dt * sigma2**2 * s_2**2) / (2 * ds2**2)
    b2 = (dt * (r - q2) * s_2) / (2 * ds2)
    d2 = - (a2[1:-1,:] + b2[1:-1,:])
    u2 = - (a2[1:-1,:] - b2[1:-1,:])
    m2 = 1 + 2*a2[1:-1,:] + dt * r / 2
    c = dt * corr * sigma1 * sigma2 * s_1 * s_2 / (8 * ds1 * ds2)
    
    for n in range(nt - 1, -1, -1):
        v_next = v
        # For x
        for j in range(1, ny):
            # explicit
            ex_y_ij = v[j,1:-1] + c[j,1:-1] * (v[j+1, 2:] - v[j-1, 2:] - v[j+1, :-2] + v[j-1, :-2])
            ex_y_ij[0] -= u1[j,0]*v[j,0]
            ex_y_ij[-1] -= d1[j,-1]*v[j,-1]
            # implicit
            ab_x = np.zeros((3, nx - 1))
            ab_x[0, 1:] = d1[j, :-1]
            ab_x[1] = m1[j, :]
            ab_x[2, :-1] = u1[j, 1:]
            v[j, 1:-1] = solve_banded((1, 1), ab_x, ex_y_ij)
        # For y
        for i in range(1, nx):
            # Explicit
            ex_x_ij = v[1:-1, i] + c[1:-1, i] * (v[2:, i+1] - v[:-2, i+1] - v[2:, i-1] + v[:-2, i-1])
            ex_x_ij[0] -= u2[0, i] * v[0, i]
            ex_x_ij[-1] -= d2[-1, i] * v[-1, i]
            # Implicit
            ab_y = np.zeros((3, ny - 1))
            ab_y[0, 1:] = d2[:-1, i]
            ab_y[1] = m2[:, i]
            ab_y[2, :-1] = u2[1:, i] 
            v[1:-1, i]  = solve_banded((1, 1), ab_y, ex_x_ij)

        # Boundary condition
        v[0, :] = 2 * v[1, :] - v[2, :]
        v[-1, :] = 2 * v[-2, :] - v[-3, :]
        v[:, 0] = 2 * v[:, 1] - v[:, 2]
        v[:, -1] = 2 * v[:, -2] - v[:, -3]
        v[0, 0] = v[0, 1] + v[1, 0] - v[1, 1]
        v[-1, 0] = v[-2, 0] + v[-1, 1] - v[-2, 1]
        v[0, -1] = v[0, -2] + v[1, -1] - v[1, -2]
        v[-1, -1] = v[-1, -2] + v[-2, -1] - v[-2, -2]

    f = RectBivariateSpline(s2_v, s1_v, v)
    f_next = RectBivariateSpline(s2_v, s1_v, v_next)
    
    # Greeks
    price = f(s2, s1)[0, 0]
    delta1 = (f(s2, s1 + ds1)[0, 0] - f(s2, s1 - ds1)[0, 0]) / (2 * ds1)
    delta2 = (f(s2 + ds2, s1)[0, 0] - f(s2 - ds2, s1)[0, 0]) / (2 * ds2)
    gamma1 = (f(s2, s1 + ds1)[0, 0] - 2 * price + f(s2, s1 - ds1)[0, 0]) / (ds1**2)
    gamma2 = (f(s2 + ds2, s1)[0, 0] - 2 * price + f(s2 - ds2, s1)[0, 0]) / (ds2**2)
    cross_gamma = (f(s2 + ds2, s1 + ds1)[0, 0] - f(s2 + ds2, s1 - ds1)[0, 0] -
                   f(s2 - ds2, s1 + ds1)[0, 0] + f(s2 - ds2, s1 - ds1)[0, 0]) / (4 * ds1 * ds2)
    theta = (f_next(s2, s1)[0, 0] - f(s2, s1)[0, 0]) / (dt * 250) # 1day theta

    return (price, delta1, delta2, gamma1, gamma2, cross_gamma, theta)