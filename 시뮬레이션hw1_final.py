# 시뮬레이션 방법론 과제1 최종본 소스코드
# 20249132 김형환

import numpy as np

def BarrierOptionsPrice(s, k, t, b, r, std, UpDown, InOut, CallPut, n=10000,m=250):
    '''
    s : underlying price at t=0
    k : strike price
    t : maturity (year)
    b : barrier price
    r : risk-free rate (annualization, 1%=0.01)
    std : standard deviation of underlying return (annualization, 1%=0.01)
    UpDown : Up is "U", Down is "D" (should be capital)
    InOut : In is "I", Out is "O" (should be capital)
    CallPut : Call is "C", Put is "P" (should be capital)
    n : number of simulation
    m : number of euler-discrete partition
    '''
    dt = t/m
    z = np.random.standard_normal( m*n ).reshape(n,m)
    underlying_path = s*np.exp((r-0.5*(std**2))*dt+std*np.sqrt(dt)*z).cumprod(axis=1)
    if UpDown=="U" and InOut=="O" :
        payoff_logic = underlying_path.max(axis=1)<=b
    elif UpDown=="U" and InOut=="I" :
        payoff_logic = underlying_path.max(axis=1)>b
    elif UpDown=="D" and InOut=="O" :
        payoff_logic = underlying_path.min(axis=1)>=b
    elif UpDown=="D" and InOut=="I" :
        payoff_logic = underlying_path.min(axis=1)<b

    if CallPut=="C" :
        plain_payoff = np.maximum(underlying_path[:,-1]-k,0)
    elif CallPut=="P" :
        plain_payoff = np.maximum(k-underlying_path[:,-1],0)    

    barrier_simulation = payoff_logic * plain_payoff * np.exp(-r*t)
    barrier_price = barrier_simulation.mean()
    return barrier_price

if __name__ == "__main__":
    import pandas as pd
    import QuantLib as ql
    
    # Analytic Solution vs. Montecarlo simulation 비교

    S = 100; r = 0.03; vol = 0.2; T = 1; K = 100; B = 120; rebate = 0
    barrierType = ql.Barrier.UpOut; optionType = ql.Option.Call

    #Barrier Option
    today = ql.Date().todaysDate(); maturity = today + ql.Period(T, ql.Years)

    payoff = ql.PlainVanillaPayoff(optionType, K)
    euExercise = ql.EuropeanExercise(maturity)
    barrierOption = ql.BarrierOption(barrierType, B, rebate, payoff, euExercise)

    #Market
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(S))
    flatRateTs = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    flatVolTs = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    bsm = ql.BlackScholesProcess(spotHandle, flatRateTs, flatVolTs)
    analyticBarrierEngine = ql.AnalyticBarrierEngine(bsm)

    #Pricing
    barrierOption.setPricingEngine(analyticBarrierEngine)
    QL_UOCprice = barrierOption.NPV()

    # Hyeonghwan Pricing (MCS)
    HH_UOCprice = BarrierOptionsPrice(S, K, T, B, r, vol, "U", "O", "C")

    print("----- Barrier Options Pricing in QuantLib vs. Montecarlo -----")
    print("Up & Out Call with S=100, K=100, B=120, T=1, Vol=0.2, r= 0.03")
    print("Number of simulation : 10000 , Number of time split(M) : 250")
    print("QuantLib price :", QL_UOCprice)
    print("Difference is", QL_UOCprice - HH_UOCprice)
    
    # N*M = Tau 제약 하에서 N에 따른 Bias, Variance, MSE 변화 추이
    
    L = 200 # 산출 신뢰도 향상을 위해각 N,M 별로 시뮬레이션 L번 반복 예정
    tau = 2**16  # 예산제약, 계산시간이 N, M과 정비례한다고 가정
    Ns, Ms, bias, var = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

    for i in range(10):
        N = 2**(i+4)
        M = int(np.round(tau / N, 0))
        y = []
        for j in range(L):
            y.append(BarrierOptionsPrice(S, K, T, B, r, vol, "U", "O", "C", n=N, m=M))
        Ns[i], Ms[i] = N, M
        bias[i] = (np.mean(y) - QL_UOCprice)**2
        var[i] = np.var(y,ddof = 1)

    result = pd.DataFrame({"N":Ns,
                        "log2(N)":np.log2(Ns),
                        "M":Ms,
                        'Bias^2':bias,
                        'Variance':var,
                        'MSE':bias+var})
    print("---------- Trend of Bias, Variance, MSE in MCS ----------")
    print("Tau = N*M = 65536, x-axis is N(log2 scale)")
    print("We can find the optimal value of N where MSE is minimized.")
    result.plot(x='log2(N)',y=['Bias^2','Variance','MSE'])