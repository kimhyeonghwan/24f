#%%
import QuantLib as ql

#Market Info.
S = 100
r = 0.03
vol = 0.2

#Product Spec.
T = 1
K = 100
B = 120
rebate = 0
barrierType = ql.Barrier.UpOut
optionType = ql.Option.Call

#Barrier Option
today = ql.Date().todaysDate()
maturity = today + ql.Period(T, ql.Years)

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
price = barrierOption.NPV()

print("Price = ", price)

# %%
