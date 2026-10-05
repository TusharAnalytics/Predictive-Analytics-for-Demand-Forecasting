# Predictive-Analytics-for-Demand-Forecasting# Predictive Analytics for Demand Forecasting

Week 4 task of a Supply Chain Analytics internship: a **conceptual demand-forecasting model** that combines historical trend, seasonality and market indicators to predict future demand.

> **Data note:** all data is **synthetic pseudo-data** created for illustration. No internal or real company data is used. Accuracy numbers only demonstrate the comparison procedure, not real-world performance.

## Objective
Design and document a practical forecasting approach for a typical FMCG manufacturer-distributor: choose methods, justify the choice, show how the model works on pseudo-data, and explain the impact on supply chain planning.

## Methods compared
| Method | Idea | Uses market indicators? |
|---|---|---|
| Seasonal Naive | Same month last year (baseline) | No |
| Holt-Winters (additive) | Smoothed level + trend + seasonal index | No |
| ARIMAX | Regression with ARIMA errors; trend, Fourier seasonality, consumer confidence, promo flag | Yes |
| **Hybrid (chosen)** | 50/50 average of Holt-Winters and ARIMAX | Yes |

Machine-learning (XGBoost) and deep-learning (LSTM, TFT) methods are discussed in the report but not run, because 36-48 monthly points are too few for them.

## Process
1. Data collection (history + public indicators such as consumer confidence and CPI)
2. Cleaning (stock-outs, outliers)
3. Exploration (trend, seasonality, correlations)
4. Feature engineering (sin/cos terms, trend, promo flag)
5. Model library (baseline, Holt-Winters, ARIMAX)
6. Back-test, compare MAE / RMSE / MAPE, build hybrid
7. Deploy to S&OP and monitor bias and error monthly

## Results (pseudo-data, 6-month back-test)
| Method | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Seasonal Naive | 12.2 | 15.5 | 4.97 |
| Holt-Winters | 4.6 | 7.7 | 1.83 |
| ARIMAX + indicators | 2.7 | 4.0 | 1.14 |
| **Hybrid** | 2.7 | 3.8 | **1.12** |

Units: thousand units per month. Errors are unrealistically low because the data was generated from the same structure the model assumes.

### Forecast Jul-Dec 2026 (hybrid, approx. 80% interval)
| Month | Forecast | Lower | Upper |
|---|---|---|---|
| Jul 2026 | 247 | 242 | 252 |
| Aug 2026 | 247 | 242 | 251 |
| Sep 2026 | 254 | 249 | 259 |
| Oct 2026 | 269 | 264 | 274 |
| Nov 2026 | 276 | 271 | 282 |
| Dec 2026 | 265 | 260 | 270 |

![Forecast](images/forecast_chart.png)
![Seasonality](images/seasonality.png)

## Impact on supply chain planning
Forecasts and their uncertainty feed safety-stock sizing, capacity and labour planning, procurement timing, warehouse and transport booking, and KPIs such as fill rate, inventory turnover and forecast accuracy. See the full report for worked examples.

## Assumptions and limitations (summary)
Past patterns continue; demand data reflects true demand; driver effects are stable; future promo/festival calendar is known. Key risks: short history, structural breaks, moving festival dates, lost sales in stock-outs, SKU-level noise.

## Repository structure
```
.
├── demand_forecasting.py   # pseudo-data, models, back-test, charts
├── requirements.txt
├── report.docx             # full Week 4 report
├── images/                 # generated charts
└── output/results.json     # generated metrics and forecasts
```

## How to run
```bash
pip install -r requirements.txt
python demand_forecasting.py
```
The seed is fixed, so results are reproducible.

## References
Box & Jenkins (1970); Winters (1960); Croston (1972); Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*; Makridakis et al. (2020, 2022) M4 and M5 competitions; Chen & Guestrin (2016) XGBoost; Lim et al. (2021) Temporal Fusion Transformers.
