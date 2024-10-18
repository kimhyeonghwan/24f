# 시뮬레이션방법론 최종과제 소스코드
# 20249132 김형환

import numpy as np
import scipy.stats as sst
import time

def mc_barrier_price(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    z = np.random.standard_normal( m*n ).reshape(n,m)
    underlying_path = s*np.exp((r-q-0.5*(sigma**2))*dt+sigma*np.sqrt(dt)*z).cumprod(axis=1)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
        
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_lhs(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    # Stratified sampling, Latin Hypercube Sampling(LHS) with dimenion=m, sample=n
    lhs = sst.qmc.LatinHypercube(d=m).random(n=n)
    z = sst.norm.ppf(lhs) # convert standard normal
    underlying_path = s*np.exp((r-q-0.5*(sigma**2))*dt+sigma*np.sqrt(dt)*z).cumprod(axis=1)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_anti(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    z = np.random.standard_normal( m*n ).reshape(n,m)
    # Antithetic variates : combine z and -z
    z = np.concatenate([z,-z], axis=0)
    underlying_path = s*np.exp((r-q-0.5*(sigma**2))*dt+sigma*np.sqrt(dt)*z).cumprod(axis=1)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
        
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_moment(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    z = np.random.standard_normal( m*n ).reshape(n,m)
    
    # Moment matching : standardization z
    z = ( z.cumsum(axis=1) - z.mean(axis=1).reshape(len(z),1) ) / z.std(axis=1,ddof=1).reshape(len(z),1)
  
    underlying_path = s * np.exp((r-q-0.5*(sigma**2))*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
        
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_lhs_anti(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    # Stratified sampling, Latin Hypercube Sampling(LHS) with dimenion=m, sample=n
    lhs = sst.qmc.LatinHypercube(d=m).random(n=n)
    z = sst.norm.ppf(lhs) # convert standard normal
    # Antithetic variates : combine z and -z
    z = np.concatenate([z,-z], axis=0)
    underlying_path = s*np.exp((r-q-0.5*(sigma**2))*dt+sigma*np.sqrt(dt)*z).cumprod(axis=1)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_lhs_moment(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    # Stratified sampling, Latin Hypercube Sampling(LHS) with dimenion=m, sample=n
    lhs = sst.qmc.LatinHypercube(d=m).random(n=n)
    z = sst.norm.ppf(lhs) # convert standard normal
    
    # Moment matching : standardization z
    z = ( z.cumsum(axis=1) - z.mean(axis=1).reshape(len(z),1) ) / z.std(axis=1,ddof=1).reshape(len(z),1)
  
    underlying_path = s * np.exp((r-q-0.5*(sigma**2))*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()
    
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_anti_moment(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m    
    z = np.random.standard_normal( m*n ).reshape(n,m)

    # Antithetic variates : combine z and -z
    z = np.concatenate([z,-z], axis=0)
    
    # Moment matching : standardization z
    z = ( z.cumsum(axis=1) - z.mean(axis=1).reshape(len(z),1) ) / z.std(axis=1,ddof=1).reshape(len(z),1)
  
    underlying_path = s * np.exp((r-q-0.5*(sigma**2))*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()

    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

def mc_barrier_price_total(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    
    # Stratified sampling, Latin Hypercube Sampling(LHS) with dimenion=m, sample=n
    lhs = sst.qmc.LatinHypercube(d=m).random(n=n)
    z = sst.norm.ppf(lhs) # convert standard normal
    
    # Antithetic variates : combine z and -z
    z = np.concatenate([z,-z], axis=0)
    
    # Moment matching : standardization z
    z = ( z.cumsum(axis=1) - z.mean(axis=1).reshape(len(z),1) ) / z.std(axis=1,ddof=1).reshape(len(z),1)
  
    underlying_path = s * np.exp((r-q-0.5*(sigma**2))*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)
    
    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv
    barrier_price = barrier_npv.mean()

    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm

# stratified sampling + importance > brownian bridge

def mc_barrier_price_is(s, k, r, q, t, sigma, option_flag, n, b, barrier_flag, m):
    start = time.time()
    dt = t/m
    z = np.random.standard_normal( m*n ).reshape(n,m)
    
    # importance sampling in barrier price & strike price
    mu1 = (np.log(b/s) - (r-q-0.5*sigma**2)*dt*(np.arange(m)+1)) / (sigma*np.sqrt(dt))
    mu2 = (np.log(b/s) - (r-q-0.5*sigma**2)*dt*t) / (sigma*np.sqrt(dt))
    z += mu
    likelihood_ratio = np.exp(-mu*z + 0.5*mu**2)
    
    z =z.cumsum(axis=1)
  
    underlying_path = s * np.exp((r-q-0.5*(sigma**2))*dt*(np.arange(m)+1) + sigma*np.sqrt(dt)*z)

    if barrier_flag=="up-out" : payoff_logic = underlying_path.max(axis=1)<=b
    elif barrier_flag=="up-in" : payoff_logic = underlying_path.max(axis=1)>b
    elif barrier_flag=="down-out" : payoff_logic = underlying_path.min(axis=1)>=b
    elif barrier_flag=="down-in" : payoff_logic = underlying_path.min(axis=1)<b
    else : return "ERROR : invalid barrier_flag"

    callOrPut = 1 if option_flag.lower()=='call' else -1
    
    plain_npv = np.maximum(callOrPut * (underlying_path[:,-1]-k),0) * np.exp(-r*t)
    barrier_npv = payoff_logic * plain_npv * likelihood_ratio
    barrier_price = barrier_npv.mean()
        
    # Control variate using plain vanilla options
    d1 = (np.log(s/k) + (r - q + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    nd1, nd2 = sst.norm.cdf(callOrPut*d1), sst.norm.cdf(callOrPut*d2)
    plain_analytic = callOrPut*(s*np.exp(-q*t)*nd1 - k*np.exp(-r*t)*nd2)

    beta = np.cov(barrier_npv,plain_npv,ddof=1)[0,1] / plain_npv.var(ddof = 1)
    barrier_priceCV = barrier_npv.mean() - beta * (plain_npv.mean() - plain_analytic)

    end = time.time()
    
    tm = end - start

    return barrier_price, barrier_priceCV, tm
