import numpy as np
import pandas as pd


def calculate_returns(prices):
    return prices.pct_change().dropna()


def portfolio_returns(returns, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return returns.dot(weights)


def value_at_risk(returns, confidence=0.95):
    return float(-np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns, confidence=0.95):
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= threshold]
    return float(-tail_losses.mean())


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def sharpe_ratio(returns, risk_free_rate=0.02):
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def market_risk_summary(returns):
    return {
        "annualized_return": float(returns.mean() * 252),
        "annualized_volatility": float(returns.std() * np.sqrt(252)),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, 0.95),
        "var_99": value_at_risk(returns, 0.99),
        "expected_shortfall_95": expected_shortfall(returns, 0.95),
        "expected_shortfall_99": expected_shortfall(returns, 0.99),
    }
