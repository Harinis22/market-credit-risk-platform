
# market-credit-risk-platform
#libraries

@"
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
streamlit>=1.35
plotly>=5.20
"@ | Set-Content requirements.txt

# Integrated Market & Credit Risk Simulation Platform

A Python-based risk analytics project that combines **Market Risk**, **Credit Risk**, **Monte Carlo Simulation**, and **Machine Learning**.  
The goal of this project is to demonstrate how financial institutions can estimate downside portfolio risk, predict borrower default probability, and calculate expected credit loss under different stress scenarios.

---

## Project Overview

This project covers two major areas of financial risk analytics:

### 1. Market Risk

Market risk measures the possibility of portfolio losses due to changes in market prices, volatility, and return behavior.

This project calculates:

- Daily portfolio returns
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Value-at-Risk, also called VaR
- Expected Shortfall
- Monte Carlo simulated portfolio losses

### 2. Credit Risk

Credit risk measures the possibility that a borrower may fail to repay a loan.

This project applies machine learning models to estimate:

- Probability of Default, also called PD
- Borrower default classification
- Credit risk bands
- Expected Credit Loss, also called ECL
- Credit losses under baseline and recession scenarios

---

## Business Use Case

Banks, insurance companies, lending firms, hedge funds, and financial institutions need to understand:

- How much money a portfolio could lose under normal and stressed markets
- Which borrowers are more likely to default
- How expected credit losses change during recession or high-interest-rate conditions
- How risk models can support better reporting, monitoring, and decision-making

This project simulates a simplified version of that real-world risk analytics workflow.

---

## Key Features

- Built a complete end-to-end risk analytics workflow in Python
- Generated synthetic market price data for a multi-asset portfolio
- Calculated market risk metrics including VaR, Expected Shortfall, volatility, Sharpe ratio, and drawdown
- Applied Monte Carlo simulation to generate 10,000 future portfolio paths
- Generated synthetic borrower-level credit data
- Trained machine learning models to predict borrower default risk
- Compared Logistic Regression and Random Forest models using ROC-AUC
- Calculated Expected Credit Loss using PD × LGD × EAD
- Performed credit stress testing under multiple economic scenarios
- Produced clean console output suitable for project demonstration and resume discussion

---

## Risk Metrics Used

| Metric | Description |
|---|---|
| Annualized Return | Estimated yearly portfolio return based on daily returns |
| Annualized Volatility | Yearly risk or uncertainty of portfolio returns |
| Sharpe Ratio | Risk-adjusted return measure |
| Maximum Drawdown | Worst peak-to-trough portfolio decline |
| Value-at-Risk | Estimated loss at a selected confidence level |
| Expected Shortfall | Average loss beyond the VaR threshold |
| Probability of Default | Estimated chance that a borrower defaults |
| Expected Credit Loss | Estimated credit loss using PD × LGD × EAD |

---

## Machine Learning Models

This project uses supervised classification models for credit default prediction.

Models included:

- Logistic Regression
- Random Forest Classifier

Model evaluation includes:

- ROC-AUC score
- Confusion matrix
- Classification report
- Predicted default probability
- Risk band assignment

---

## Monte Carlo Simulation

Monte Carlo simulation is used to generate thousands of possible future portfolio paths.

The simulation estimates:

- Mean final portfolio value
- Mean profit or loss
- Probability of portfolio loss
- Simulated 95% VaR
- Simulated 99% VaR
- Worst 1% average loss

This helps estimate portfolio tail risk under uncertainty.

---

## Credit Loss Stress Testing

Expected Credit Loss is calculated using:

```text
Expected Credit Loss = Probability of Default × Loss Given Default × Exposure at Default

## Interactive Dashboard

This project includes a Streamlit dashboard for market risk, Monte Carlo simulation, credit risk modeling, and expected credit loss stress testing.

Run the dashboard locally:

```bash
streamlit run dashboard.py

##Dashboard views include:

Market risk summary
Portfolio cumulative return
Daily return distribution
Monte Carlo portfolio paths
Credit model ROC-AUC
Probability of default distribution
Confusion matrix
Expected Credit Loss stress scenarios


---

# Step 5: Push dashboard to GitHub

```powershell
git status
git add .
git commit -m "Add Streamlit risk dashboard"
git pull --rebase origin main
=======
# market-credit-risk-platform
#libraries

@"
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
streamlit>=1.35
plotly>=5.20
"@ | Set-Content requirements.txt

# Integrated Market & Credit Risk Simulation Platform

A Python-based risk analytics project that combines **Market Risk**, **Credit Risk**, **Monte Carlo Simulation**, and **Machine Learning**.  
The goal of this project is to demonstrate how financial institutions can estimate downside portfolio risk, predict borrower default probability, and calculate expected credit loss under different stress scenarios.

---

## Project Overview

This project covers two major areas of financial risk analytics:

### 1. Market Risk

Market risk measures the possibility of portfolio losses due to changes in market prices, volatility, and return behavior.

This project calculates:

- Daily portfolio returns
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Value-at-Risk, also called VaR
- Expected Shortfall
- Monte Carlo simulated portfolio losses

### 2. Credit Risk

Credit risk measures the possibility that a borrower may fail to repay a loan.

This project applies machine learning models to estimate:

- Probability of Default, also called PD
- Borrower default classification
- Credit risk bands
- Expected Credit Loss, also called ECL
- Credit losses under baseline and recession scenarios

---

## Business Use Case

Banks, insurance companies, lending firms, hedge funds, and financial institutions need to understand:

- How much money a portfolio could lose under normal and stressed markets
- Which borrowers are more likely to default
- How expected credit losses change during recession or high-interest-rate conditions
- How risk models can support better reporting, monitoring, and decision-making

This project simulates a simplified version of that real-world risk analytics workflow.

---

## Key Features

- Built a complete end-to-end risk analytics workflow in Python
- Generated synthetic market price data for a multi-asset portfolio
- Calculated market risk metrics including VaR, Expected Shortfall, volatility, Sharpe ratio, and drawdown
- Applied Monte Carlo simulation to generate 10,000 future portfolio paths
- Generated synthetic borrower-level credit data
- Trained machine learning models to predict borrower default risk
- Compared Logistic Regression and Random Forest models using ROC-AUC
- Calculated Expected Credit Loss using PD × LGD × EAD
- Performed credit stress testing under multiple economic scenarios
- Produced clean console output suitable for project demonstration and resume discussion

---

## Risk Metrics Used

| Metric | Description |
|---|---|
| Annualized Return | Estimated yearly portfolio return based on daily returns |
| Annualized Volatility | Yearly risk or uncertainty of portfolio returns |
| Sharpe Ratio | Risk-adjusted return measure |
| Maximum Drawdown | Worst peak-to-trough portfolio decline |
| Value-at-Risk | Estimated loss at a selected confidence level |
| Expected Shortfall | Average loss beyond the VaR threshold |
| Probability of Default | Estimated chance that a borrower defaults |
| Expected Credit Loss | Estimated credit loss using PD × LGD × EAD |

---

## Machine Learning Models

This project uses supervised classification models for credit default prediction.

Models included:

- Logistic Regression
- Random Forest Classifier

Model evaluation includes:

- ROC-AUC score
- Confusion matrix
- Classification report
- Predicted default probability
- Risk band assignment

---

## Monte Carlo Simulation

Monte Carlo simulation is used to generate thousands of possible future portfolio paths.

The simulation estimates:

- Mean final portfolio value
- Mean profit or loss
- Probability of portfolio loss
- Simulated 95% VaR
- Simulated 99% VaR
- Worst 1% average loss

This helps estimate portfolio tail risk under uncertainty.

---

## Credit Loss Stress Testing

Expected Credit Loss is calculated using:

```text
Expected Credit Loss = Probability of Default × Loss Given Default × Exposure at Default

## Interactive Dashboard

This project includes a Streamlit dashboard for market risk, Monte Carlo simulation, credit risk modeling, and expected credit loss stress testing.

Run the dashboard locally:

```bash
python -m pip install streamlit plotly
streamlit run dashboard.py

##Dashboard views include:

Market risk summary
Portfolio cumulative return
Daily return distribution
Monte Carlo portfolio paths
Credit model ROC-AUC
Probability of default distribution
Confusion matrix
Expected Credit Loss stress scenarios


---

# Step 5: Push dashboard to GitHub

```powershell
git status
git add .
git commit -m "Add Streamlit risk dashboard"
git pull --rebase origin main
 (Add Streamlit dashboard)
git push origin main
