"""
Week 4 - Predictive Analytics for Demand Forecasting (conceptual model demo)

Generates ILLUSTRATIVE pseudo-data (no real company data), then compares
Seasonal Naive, Holt-Winters, ARIMAX (with market indicators) and a Hybrid
average, back-tests on 6 held-out months and forecasts Jul-Dec 2026.

Run:  python demand_forecasting.py
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

rng = np.random.default_rng(42)
idx = pd.date_range("2023-01-01", periods=48, freq="MS")  # Jan23 - Dec26
t = np.arange(48)
season_pat = np.array([-8,-12,-5,0,4,7,5,3,6,12,22,18])  # festive peak Nov/Dec, dip Feb
seas = np.array([season_pat[m.month-1] for m in idx])
cci = 100 + 3*np.sin(t/7) + rng.normal(0,1.2,48)          # consumer confidence index
promo = np.zeros(48); 
for m in [5,10,16,22,29,34,40,45]: promo[m]=1               # promo months
base = 200 + 1.1*t
demand = base + seas + 1.6*(cci-100) + 14*promo + rng.normal(0,4,48)
demand = np.round(demand).astype(int)
df = pd.DataFrame({"date":idx,"demand":demand,"cci":np.round(cci,1),"promo":promo.astype(int)})
train_n, test_n = 36, 6           # train Jan23-Dec25, test Jan26-Jun26
fut_n = 6                         # forecast Jul26-Dec26
tr = df.iloc[:train_n]; te = df.iloc[train_n:train_n+test_n]; fu = df.iloc[train_n+test_n:]

def four(t, k=2):
    cols={}
    for j in range(1,k+1):
        cols[f"s{j}"]=np.sin(2*np.pi*j*t/12); cols[f"c{j}"]=np.cos(2*np.pi*j*t/12)
    return pd.DataFrame(cols)
def exog(d, tt):
    X = four(tt); X["cci"]=d["cci"].values; X["promo"]=d["promo"].values; X["trend"]=tt
    return X

def mape(a,f): return float(np.mean(np.abs((a-f)/a))*100)
def rmse(a,f): return float(np.sqrt(np.mean((a-f)**2)))
def mae(a,f): return float(np.mean(np.abs(a-f)))

def fit_all(trn, h, trn_t, fut_df, fut_t):
    y = trn["demand"].values.astype(float)
    out={}
    # seasonal naive
    out["Seasonal Naive"] = np.array([y[-12+(i%12)] for i in range(h)])
    # Holt-Winters additive
    hw = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
    out["Holt-Winters"] = hw.forecast(h)
    # Regression w/ ARIMA errors + market indicators
    sx = SARIMAX(y, exog=exog(trn,trn_t).values, order=(1,0,0), trend="c").fit(disp=False)
    out["ARIMAX + indicators"] = sx.forecast(h, exog=exog(fut_df,fut_t).values)
    out["Hybrid (average)"] = (out["Holt-Winters"]+out["ARIMAX + indicators"])/2
    return out, hw, sx

preds, hw, sx = fit_all(tr, test_n, t[:train_n], te, t[train_n:train_n+test_n])
actual = te["demand"].values.astype(float)
metrics={}
for k,v in preds.items():
    metrics[k]={"MAE":mae(actual,v),"RMSE":rmse(actual,v),"MAPE":mape(actual,v)}
print(json.dumps(metrics,indent=1))

# forward forecast: refit on 42 months, future exog assumed
trn2 = df.iloc[:42]
preds2, _, _ = fit_all(trn2, fut_n, t[:42], fu, t[42:48])
fc = preds2["Hybrid (average)"]
# interval from holdout RMSE of hybrid
r = metrics["Hybrid (average)"]["RMSE"]
lo = fc-1.28*r*np.sqrt(1+np.arange(fut_n)*0.05); hi = fc+1.28*r*np.sqrt(1+np.arange(fut_n)*0.05)

# chart
fig,ax=plt.subplots(figsize=(9,4.2),dpi=150)
ax.plot(df["date"][:42],df["demand"][:42],color="#1F3A5F",lw=1.8,label="Historical demand (pseudo-data)")
ax.plot(te["date"],preds["Hybrid (average)"],color="#E07B00",lw=2,ls="--",marker="o",ms=4,label="Hybrid back-test forecast")
ax.plot(fu["date"],fc,color="#2E8B57",lw=2,marker="s",ms=4,label="Hybrid forecast Jul-Dec 2026")
ax.fill_between(fu["date"],lo,hi,color="#2E8B57",alpha=0.18,label="~80% prediction band")
ax.axvline(pd.Timestamp("2025-12-16"),color="grey",ls=":",lw=1); ax.text(pd.Timestamp("2025-12-22"),df["demand"].min()+2,"back-test window",fontsize=8,color="grey")
ax.set_ylabel("Demand (000 units per month)"); ax.set_title("Monthly demand: history, back-test and forward forecast",fontsize=11)
ax.grid(alpha=0.25); ax.legend(fontsize=8,loc="upper left"); 
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("images/forecast_chart.png")

# second chart: seasonal index
fig,ax=plt.subplots(figsize=(6.5,3.2),dpi=150)
mm = df.iloc[:36].copy(); mm["detr"]=mm["demand"]-mm["demand"].rolling(12,center=True,min_periods=6).mean()
si = mm.groupby(mm["date"].dt.month)["detr"].mean()
ax.bar(["J","F","M","A","M","J","J","A","S","O","N","D"],si.values,color=["#C0392B" if v<0 else "#1F3A5F" for v in si.values])
ax.axhline(0,color="black",lw=0.6); ax.set_ylabel("000 units vs. trend"); ax.set_title("Estimated seasonal effect by month (pseudo-data)",fontsize=10)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("images/seasonality.png")

out={"metrics":metrics,
 "hist":[[d.strftime("%b %Y"),int(a),float(c),int(p)] for d,a,c,p in zip(df["date"][:42],df["demand"][:42],df["cci"][:42],df["promo"][:42])],
 "test":[[d.strftime("%b %Y"),int(a)]+[round(float(preds[k][i])) for k in ["Seasonal Naive","Holt-Winters","ARIMAX + indicators","Hybrid (average)"]] for i,(d,a) in enumerate(zip(te["date"],te["demand"]))],
 "future":[[d.strftime("%b %Y"),round(float(f)),round(float(l)),round(float(h)),float(c),int(p)] for d,f,l,h,c,p in zip(fu["date"],fc,lo,hi,fu["cci"],fu["promo"])],
 "coef":{k:float(v) for k,v in zip(["const"]+list(exog(tr,t[:36]).columns)+["ar1","sigma2"], sx.params)},
 "si":[round(float(v),1) for v in si.values]}
json.dump(out,open("output/results.json","w"),indent=1)
print(json.dumps(out["future"])); print(out["coef"]); print(out["si"])
