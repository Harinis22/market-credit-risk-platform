# market-credit-risk-platform

@"
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
"@ | Set-Content requirements.txt
@"
# Integrated Market & Credit Risk Simulation Platform

Python project for market risk and credit risk analytics using Monte Carlo simulation and machine learning.

## Business Goal

This project combines two major risk analytics areas:

1. Market Risk: estimate downside loss for a portfolio using VaR, Expected Shortfall, volatility, drawdown, and Monte Carlo simulation.
2. Credit Risk: predict borrower default probability using ML classification and estimate Expected Credit Loss using PD, LGD, and EAD.

## Main Features

- Historical and synthetic portfolio return generation
- Market risk metrics: volatility, Sharpe ratio, maximum drawdown, VaR, Expected Shortfall
- Monte Carlo simulation for future portfolio paths
- Credit default prediction using Logistic Regression and Random Forest
- Credit loss simulation using PD x LGD x EAD
- Baseline, high-rate, mild recession, and severe recession stress scenarios

## Project Structure

```text
market-credit-risk-platform/
├── run_project.py
├── requirements.txt
├── src/
│   ├── market_risk.py
│   ├── credit_risk.py
│   └── simulation.py
├── data/
│   └── README.md
└── outputs/
