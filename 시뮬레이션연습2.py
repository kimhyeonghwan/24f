# 시뮬레이션방법론 최종과제 소스코드
# 20249132 김형환

import numpy as np
import scipy.stats as sst
import time

def mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    
    dt = t/m
    z = np.random.randn(n,m).cumsum(axis=1)
    paths = s*np.exp((r-q-0.5*sigma**2)*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)
    
    knock = np.where(barrier_flag.split("-")[0].lower()=='up', paths.max(axis=1)>=b, paths.min(axis=1)<=b)
    knock = np.where(barrier_flag.split('-')[1].lower()=="out", ~knock, knock)
    type = np.where(option_flag.lower()=='call',1, -1)
    
    plain_npv = np.maximum(type*(paths[:,-1]-k), 0) * np.exp(-r*t)
    barrier_npv = knock * plain_npv
    barrier_price = barrier_npv.mean()
    se = barrier_npv.std(ddof=1) / np.sqrt(n)
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(type*d1), sst.norm.cdf(type*d2)
    plain_bsprice = type*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
    
    cov_npv = np.cov(barrier_npv,plain_npv,ddof=1)
    beta = np.where(cov_npv[1,1]==0,0,cov_npv[0,1] / cov_npv[1,1])
    barrier_CVnpv = barrier_npv - beta * (plain_npv - plain_bsprice)
    barrier_CVprice = barrier_CVnpv.mean()
    CVse = barrier_CVnpv.std(ddof=1) / np.sqrt(n)
    
    end = time.time()
    tm = end-start
    return barrier_price, se,  barrier_CVprice, CVse ,tm

def mc_barrier_price2(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    
    dt = t/m
    dts = np.arange(dt, t+dt, dt)
    # Stratified sampling, z_t makes underlying dist. at T & z makes brownian bridge
    z_t = sst.norm.ppf((np.arange(n) + np.random.uniform(0,1,n)) / n)
    z = np.random.randn(n,m)
    
    # Moment matching, standardization z_t
    # z_t = (z_t - z_t.mean()) / z_t.std(ddof=1)
    
    # Antithetic variate
    # z_t, z = np.concatenate([z_t, -z_t], axis=0), np.concatenate([z, -z], axis=0)
    
    w_t, w = z_t * np.sqrt(t), z.cumsum(axis=1) * np.sqrt(dt) # winner process
    bridge = dts * ((w_t- w[:,-1]).reshape(len(w),1) + w / dts) # brownian bridge
    paths = s*np.exp((r-q-0.5*sigma**2)*dts + sigma*bridge) # underlying price path
    
    knock = np.where(barrier_flag.split("-")[0].lower()=='up', paths.max(axis=1)>=b, paths.min(axis=1)<=b)
    knock = np.where(barrier_flag.split('-')[1].lower()=="out", ~knock, knock)
    type = np.where(option_flag.lower()=='call',1, -1)
    
    plain_npv = np.maximum(type*(paths[:,-1]-k), 0) * np.exp(-r*t)
    barrier_npv = knock * plain_npv
    barrier_price = barrier_npv.mean()
    se = barrier_npv.std(ddof=1) / np.sqrt(n)
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(type*d1), sst.norm.cdf(type*d2)
    plain_bsprice = type*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
    
    cov_npv = np.cov(barrier_npv,plain_npv,ddof=1)
    beta = np.where(cov_npv[1,1]==0,0,cov_npv[0,1] / cov_npv[1,1])
    barrier_CVnpv = barrier_npv - beta * (plain_npv - plain_bsprice)
    barrier_CVprice = barrier_CVnpv.mean()
    CVse = barrier_CVnpv.std(ddof=1) / np.sqrt(n)
    
    end = time.time()
    tm = end-start
    return barrier_price, se,  barrier_CVprice, CVse ,tm

def mc_barrier_price_final(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    
    dt = t/m
    dts = np.arange(dt, t+dt, dt)
    # Stratified sampling, z_t makes underlying dist. at T & z makes brownian bridge
    z_t = sst.norm.ppf((np.arange(n) + np.random.uniform(0,1,n)) / n)
    z = np.random.randn(n,m)
    
    # Moment matching, additive in z_t
    z_t = z_t - z_t.mean()
    
    # Antithetic variate
    z_t, z = np.concatenate([z_t, -z_t], axis=0), np.concatenate([z, -z], axis=0)
    
    # Importance sampling at z_t, using strike price = K
    mu = (np.log(k/s) - (r-q-0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    z_t = z_t + mu
    likelihood_ratio = np.exp(-mu*z_t + 0.5*mu**2)
    
    w_t, w = z_t * np.sqrt(t), z.cumsum(axis=1) * np.sqrt(dt) # winner process
    bridge = dts * ((w_t- w[:,-1]).reshape(len(w),1) + w / dts) # brownian bridge
    paths = s*np.exp((r-q-0.5*sigma**2)*dts + sigma*bridge) # underlying price path
    
    knock = np.where(barrier_flag.split("-")[0].lower()=='up', paths.max(axis=1)>=b, paths.min(axis=1)<=b)
    knock = np.where(barrier_flag.split('-')[1].lower()=="out", ~knock, knock)
    type = np.where(option_flag.lower()=='call',1, -1)
    
    plain_npv = np.maximum(type*(paths[:,-1]-k), 0) * np.exp(-r*t) * likelihood_ratio
    barrier_npv = knock * plain_npv
    barrier_price = barrier_npv.mean()
    se = barrier_npv.std(ddof=1) / np.sqrt(n)
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(type*d1), sst.norm.cdf(type*d2)
    plain_bsprice = type*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)
    
    cov_npv = np.cov(barrier_npv,plain_npv,ddof=1)
    beta = np.where(cov_npv[1,1]==0,0,cov_npv[0,1] / cov_npv[1,1])
    barrier_CVnpv = barrier_npv - beta * (plain_npv - plain_bsprice)
    barrier_CVprice = barrier_CVnpv.mean()
    CVse = barrier_CVnpv.std(ddof=1) / np.sqrt(n)
    
    end = time.time()
    tm = end-start
    return barrier_price, se,  barrier_CVprice, CVse ,tm
