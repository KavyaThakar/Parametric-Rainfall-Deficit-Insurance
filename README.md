# Parametric Rainfall-Deficit Insurance: Saurashtra Kharif Agriculture

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![SciPy](https://img.shields.io/badge/SciPy-Advanced%20Statistics-8CAAE6.svg)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualizations-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Source](https://img.shields.io/badge/Data-NASA%20POWER%20API-orange.svg)](https://power.larc.nasa.gov/)

An end-to-end quantitative underwriting, actuarial pricing, and basis-risk monitoring platform for **parametric weather-index insurance**. Designed for rainfed agrarian economies, specifically the drought-prone **Saurashtra region of Gujarat, India** (covering Rajkot, Junagadh, Jamnagar, Amreli, and Surendranagar) during the critical **Kharif crop season (June–September)**.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [System Architecture](#system-architecture)
- [Actuarial & Mathematical Methodology](#actuarial--mathematical-methodology)
  - [1. Data Pipeline & Index Construction](#1-data-pipeline--index-construction)
  - [2. Parametric Probability Density Fitting](#2-parametric-probability-density-fitting)
  - [3. Contract Structure & Payout Function](#3-contract-structure--payout-function)
  - [4. Pure Premium & Gross Underwriting Pricing](#4-pure-premium--gross-underwriting-pricing)
  - [5. Tail Risk Metrics (VaR & TVaR)](#5-tail-risk-metrics-var--tvar)
  - [6. Multi-Peril Gaussian Copula Engine](#6-multi-peril-gaussian-copula-engine)
  - [7. Agronomic Basis Risk (FAO-33 Soil-Water Balance)](#7-agronomic-basis-risk-fao-33-soil-water-balance)
  - [8. Climate Trend & Non-Stationarity Testing](#8-climate-trend--non-stationarity-testing)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
  - [1. Basic Pricing CLI](#1-basic-pricing-cli)
  - [2. Advanced Analytics CLI](#2-advanced-analytics-cli)
  - [3. Interactive Streamlit Dashboard](#3-interactive-streamlit-dashboard)
- [Meteorological Coverage](#meteorological-coverage)
- [Validation & Backtesting](#validation--backtesting)
- [Industry & Academic Limitations](#industry--academic-limitations)
- [License](#license)

---

## Executive Overview

Traditional indemnity-based crop insurance suffers from high administrative costs, protracted claims settlement processes (often exceeding 6–12 months), moral hazard, and asymmetric information. **Parametric (index-based) insurance** solves these bottlenecks by pegging claim payouts strictly to objective, satellite-derived meteorological triggers rather than individual farm-level loss assessments.

This repository implements a production-grade actuarial framework that:
1. Ingests 34+ years (1991–2024) of daily corrected precipitation data (`PRECTOTCORR`) from the **NASA POWER API**.
2. Fits heavy-tailed parametric distributions (Gamma, Lognormal, Weibull) selected via **Akaike Information Criterion (AIC)** and validated via **Kolmogorov-Smirnov (KS)** hypothesis tests.
3. Prices non-linear indemnity structures using **200,000-path Monte Carlo simulations** incorporating standard deviation risk loading and administrative expense margins.
4. Extends underwriting to a **3-peril joint risk structure** (cumulative deficit, consecutive dry spells, and torrential excess rainfall) coupled through a **Gaussian Copula**.
5. Measures and optimizes contract parameters against an **agronomic soil-water deficit proxy (FAO-33 yield response)** to systematically minimize basis risk.

---

## System Architecture

```mermaid
flowchart TD
    A[NASA POWER API / Local Cache] -->|Daily Precipitation 1991-2024| B[Data Cleaning & Ingestion]
    B --> C[Seasonal Index Construction<br/>Kharif: Jun-Sep]
    
    subgraph Actuarial Underwriting Pipeline
        C --> D[Distribution Fitting<br/>Gamma | Lognormal | Weibull]
        D --> E[Goodness of Fit<br/>AIC Ranking & KS-Test]
        E --> F[200k Monte Carlo Simulation]
        F --> G[Actuarial Pricing Engine<br/>Burn Cost + Risk Load + Expense Load]
        G --> H[Solvency & Tail Risk<br/>VaR 99% & TVaR 99%]
    end
    
    subgraph Advanced Multi-Peril Engine
        C --> I[Multi-Index Feature Extraction<br/>Deficit | Dry Spells | 3-Day Excess]
        I --> J[Joint Dependence Modeling<br/>Gaussian Copula Simulation]
        J --> K[FAO-33 Soil Moisture Yield Proxy]
        K --> L[Basis Risk Evaluation & Grid Search Optimization]
        I --> M[Mann-Kendall Trend & Sen's Slope Test]
        I --> N[Walk-Forward Out-of-Sample Backtest]
    end
    
    G --> O[Interactive Streamlit Dashboard]
    H --> O
    L --> O
    N --> O
```

---

## Actuarial & Mathematical Methodology

### 1. Data Pipeline & Index Construction
Rainfall is collected across 5 meteorological stations in Gujarat, India. The primary index $R_y$ for Kharif season of year $y$ aggregates daily precipitation $P_{t}$ over the crop growth window:
$$R_y = \sum_{t \in \text{Kharif}} P_{t}, \quad \text{where } \text{Kharif} = \{\text{June 1 to September 30}\}$$

### 2. Parametric Probability Density Fitting
To eliminate sample overfitting from purely historical burn costs, marginal seasonal rainfall is fitted across candidate parametric families:
- **Gamma Distribution**:
  $$f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0$$
- **Lognormal Distribution**:
  $$f(x; \mu, \sigma) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$$
- **Weibull (Min) Distribution**:
  $$f(x; k, \lambda) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k - 1} e^{-(x/\lambda)^k}$$

The optimal distribution is selected using the Akaike Information Criterion:
$$\text{AIC} = 2k - 2\ln(\hat{L})$$
and goodness-of-fit is verified through the two-sample Kolmogorov-Smirnov test statistic $D = \sup_x |F_n(x) - F_0(x)|$.

### 3. Contract Structure & Payout Function
The insurance policy specifies a **Strike** threshold $K_{\text{strike}}$ (point of first payout) and an **Exit** threshold $K_{\text{exit}}$ (point of total crop failure / maximum indemnity payout):

$$\text{Payout}(R) = \begin{cases} 
0 & \text{if } R \ge K_{\text{strike}} \\
\text{Limit} \times \left(\frac{K_{\text{strike}} - R}{K_{\text{strike}} - K_{\text{exit}}}\right) & \text{if } K_{\text{exit}} < R < K_{\text{strike}} \\
\text{Limit} & \text{if } R \le K_{\text{exit}} 
\end{cases}$$

where $\text{Limit}$ denotes the total Sum Insured (INR).

### 4. Pure Premium & Gross Underwriting Pricing
Using $N = 200,000$ simulated realizations $R^{(i)} \sim \hat{F}_R$:
- **Pure Burn Cost (Expected Loss)**:
  $$\mathbb{E}[\text{Loss}] = \frac{1}{N} \sum_{i=1}^N \text{Payout}(R^{(i)})$$
- **Risk Loading**: To buffer against severe parameter uncertainty and underwriting volatility, a standard deviation loading $k$ (typically $0.20 \le k \le 0.35$) is applied:
  $$\text{Risk Margin} = k \cdot \sigma[\text{Loss}]$$
- **Gross Commercial Premium**: Incorporating commercial insurer expense ratio $e \in [0, 1)$ (brokerage, reinsurance margin, distribution, operational overhead):
  $$\text{Gross Premium} = \frac{\mathbb{E}[\text{Loss}] + k \cdot \sigma[\text{Loss}]}{1 - e}$$

### 5. Tail Risk Metrics (VaR & TVaR)
To comply with Solvency II and insurance regulatory solvency standards:
- **Value at Risk (VaR 99%)**:
  $$\text{VaR}_{0.99} = \inf \{ l \in \mathbb{R} : \mathbb{P}(\text{Loss} \le l) \ge 0.99 \}$$
- **Tail Value at Risk / Expected Shortfall (TVaR 99%)**:
  $$\text{TVaR}_{0.99} = \mathbb{E}[\text{Loss} \mid \text{Loss} > \text{VaR}_{0.99}]$$

### 6. Multi-Peril Gaussian Copula Engine
Crop yields do not fail solely from total seasonal deficit; intraseasonal dry spells and waterlogging are equally catastrophic. We construct a 3-dimensional peril vector:
1. **Deficit Index**: Cumulative Kharif rainfall (mm).
2. **Dry Spell Index**: Maximum consecutive days with rainfall $< 2.5\text{ mm/day}$.
3. **Excess Rain Index**: Maximum rolling 3-day precipitation sum (mm).

The joint distribution is simulated via a Gaussian Copula:
$$C_{\Sigma}^{\text{Gauss}}(u_1, u_2, u_3) = \Phi_{\Sigma}\left(\Phi^{-1}(u_1), \Phi^{-1}(u_2), \Phi^{-1}(u_3)\right)$$
where $\Phi$ is the standard univariate Gaussian CDF, and $\Sigma$ is the empirical rank correlation matrix calibrated across historical observations.

### 7. Agronomic Basis Risk (FAO-33 Soil-Water Balance)
Basis risk occurs when an insured farmer suffers yield loss without receiving an insurance payout (false negative / Type II error), or vice versa (false alarm / Type I error).

We benchmark the contract against a daily bucket soil moisture model parameterized following **FAO Irrigation and Drainage Paper No. 33**:
$$S_t = \min\left(S_{\max}, \max\left(0, S_{t-1} + P_t - \text{ET}_{c,t}\right)\right)$$
where $\text{ET}_{c,t} = K_{c,t} \cdot \text{ET}_0$. The relative yield reduction is calculated via:
$$\left(1 - \frac{Y_a}{Y_m}\right) = K_y \left(1 - \frac{\text{ET}_a}{\text{ET}_c}\right)$$
The grid search optimizer calibrates $(K_{\text{strike}}, K_{\text{exit}})$ to minimize:
$$\text{RMSE}_{\text{basis}} = \sqrt{\frac{1}{T}\sum_{t=1}^T \left(\frac{\text{Payout}_t}{\text{Limit}} - \text{YieldLoss}_t\right)^2}$$
under a user-defined premium constraint.

### 8. Climate Trend & Non-Stationarity Testing
Actuarial pricing relies on the stationarity hypothesis. The non-parametric **Mann-Kendall test** calculates the sign statistic $S$:
$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(x_j - x_k)$$
coupled with **Sen's Slope estimator** to verify whether secular climate change invalidates historical burn-cost benchmarks.

---

## Key Features

- **Automated Weather Ingestion**: Integrated with NASA Langley Research Center POWER REST API with automated fallback to synthetic monsoon data if offline.
- **Statistical Model Selection**: Automated goodness-of-fit test suite evaluating Gamma, Lognormal, Weibull, and Gumbel families.
- **Monte Carlo Engine**: Vectorized NumPy engine executing 200,000 paths in under 200 milliseconds.
- **Basis Risk Diagnostics**: True Hit Rate, False Alarm Rate, and Missed Drought ratios measured against the Standardized Precipitation Index ($\text{SPI} \le -1.0$).
- **Multi-Site Spatial Correlator**: Saurashtra regional correlation matrix across Rajkot, Junagadh, Jamnagar, Amreli, and Surendranagar.
- **Walk-Forward Validation**: Out-of-sample backtest simulating historical underwriter balance sheets year-by-year without lookahead bias.
- **Interactive UI**: High-fidelity Streamlit dashboard with real-time parametric curve manipulation and Plotly graphics.

---

## Directory Structure

```text
Parametric-Rainfall-Deficit-Insurance/
├── .gitignore                     # Git exclusion rules (cache, venv, temporary files)
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive project documentation & theory
├── pricing.py                     # Single-site actuarial engine & distribution fitting
├── advanced.py                    # Multi-site, copula, FAO-33 basis risk & backtesting
├── app.py                         # Production Streamlit web application
├── rainfall_daily.csv             # Baseline daily precipitation cache (Rajkot, 1991-2024)
├── rainfall_Amreli.csv            # NASA POWER daily records: Amreli District
├── rainfall_Jamnagar.csv          # NASA POWER daily records: Jamnagar District
├── rainfall_Junagadh.csv          # NASA POWER daily records: Junagadh District
├── rainfall_Rajkot.csv            # NASA POWER daily records: Rajkot District
└── rainfall_Surendranagar.csv     # NASA POWER daily records: Surendranagar District
```

---

## Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- `git` version control

### 1. Clone the Repository
```bash
git clone https://github.com/KavyaThakar/Parametric-Rainfall-Deficit-Insurance.git
cd Parametric-Rainfall-Deficit-Insurance
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution & Usage

### 1. Basic Pricing CLI
Runs statistical fitting on Rajkot historical Kharif precipitation, ranks distributions by AIC, computes contract burn rates, and generates an actuarial summary:

```bash
python pricing.py
```

*Example Output Snippet:*
```text
Distribution fits (AIC, lower = better):
  gamma       AIC = 422.14   KS stat = 0.082 (p = 0.941)
  lognormal   AIC = 424.31   KS stat = 0.091 (p = 0.887)
  weibull_min AIC = 425.80   KS stat = 0.104 (p = 0.762)

Contract Terms:
  Strike : 352.4 mm (20th percentile)
  Exit   : 231.8 mm (5th percentile)
  Limit  : INR 100,000

Pricing (200k Monte Carlo):
  Expected Loss : INR 4,120
  Loss SD       : INR 14,850
  Gross Premium : INR 9,790 (Rate on Line: 9.79%)
  Loss Ratio    : 42.08%
  VaR 99%       : INR 78,420
  TVaR 99%      : INR 94,110
```

### 2. Advanced Analytics CLI
Executes the Gaussian copula multi-peril engine, bootstrap confidence intervals, Mann-Kendall trend tests, walk-forward validation, and the FAO-33 basis risk grid search optimizer:

```bash
python advanced.py
```

### 3. Interactive Streamlit Dashboard
Launches the full interactive underwriting studio:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser. The dashboard provides:
- **Contract Sliders**: Real-time adjustment of strike percentiles, exit levels, risk loadings, and expense ratios.
- **Actuarial Visualizers**: Empirical vs fitted probability density functions (PDF) and cumulative distribution functions (CDF).
- **Underwriting Solvency**: Live calculation of VaR (99%), TVaR (99%), and Loss Ratio.
- **Basis Risk Matrix**: Spatial cross-district correlation heatmaps and drought trigger hit rates.
- **In-Season Monitor**: Real-time progress tracker comparing current season-to-date precipitation against policy triggers.

---

## Meteorological Coverage

| District | Latitude (°N) | Longitude (°E) | Agro-Climatic Zone | Primary Kharif Crops |
| :--- | :---: | :---: | :--- | :--- |
| **Rajkot** | 22.30 | 70.80 | North Saurashtra | Groundnut, Cotton, Sesame |
| **Junagadh** | 21.52 | 70.46 | South Saurashtra | Groundnut, Cotton, Castor |
| **Jamnagar** | 22.47 | 70.07 | North Saurashtra | Groundnut, Bajra, Cotton |
| **Amreli** | 21.60 | 71.22 | South Saurashtra | Cotton, Groundnut, Sesame |
| **Surendranagar** | 22.73 | 71.65 | North Saurashtra | Cotton, Bajra, Sesame |

*Data attributes:* 
- Variable: Corrected Precipitation (`PRECTOTCORR`) in mm/day.
- Temporal coverage: 1991-01-01 to 2024-12-31 (12,419 daily observations per district).

---

## Validation & Backtesting

1. **Trigger Accuracy vs SPI**: Validated against Standardized Precipitation Index ($\text{SPI} \le -1.0$) drought events. Contracts achieve $\ge 85\%$ hit rates on moderate-to-severe drought occurrences.
2. **Walk-Forward Out-of-Sample Testing**: The pricing model is trained on years $[1991, T-1]$ and tested on year $T$ recursively across 15 historical seasons. The realized long-term loss ratio converges within the actuarial target band of $65\% \pm 12\%$.
3. **Bootstrap Uncertainty**: 1,000 bootstrap resamples generate tight $95\%$ confidence bounds around pure premium rates, guarding against small-sample parameter volatility.

---

## Industry & Academic Limitations

When presenting or deploying this model for commercial or regulatory underwriting, acknowledge the following domain constraints:
1. **Agronomic Calibration**: The FAO-33 bucket yield proxy applies standardized reference evapotranspiration ($\text{ET}_0$) and representative crop coefficients ($K_c$). Production commercial underwriting requires local calibration against district crop cutting experiments (CCE) from the Directorate of Agriculture.
2. **Tail Dependence**: The standard Gaussian Copula exhibits asymptotic tail independence. In scenarios with extreme compound catastrophes (e.g., severe regional El Niño), Archimedean or Student's $t$-copulas should be tested to capture heavy tail dependence.
3. **In-Sample Optimization**: Grid search optimization for trigger thresholds must be validated on out-of-sample test splits to avoid overfitting to specific historical drought sequences.
4. **Regulatory Filings**: Expense loadings, capital charges, and profit margins are illustrative and must conform to domestic insurance regulator guidelines (e.g., IRDAI in India).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
