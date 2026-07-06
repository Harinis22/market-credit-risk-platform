import numpy as np
import pandas as pd


def calculate_returns(prices):
    """Calculate daily percentage returns from a price DataFrame."""
    returns = prices.pct_change().dropna()
    return returns


def portfolio_returns(returns, weights):
    """Calculate weighted portfolio returns."""
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    if returns.shape[1] != len(weights):
        raise ValueError(
            f"Number of return columns ({returns.shape[1]}) must match number of weights ({len(weights)})."
        )

    return returns.dot(weights)


def value_at_risk(returns, confidence_level=0.95):
    """Historical Value-at-Risk as a negative return threshold."""
    return float(np.percentile(returns, (1 - confidence_level) * 100))


def expected_shortfall(returns, confidence_level=0.95):
    """Average loss beyond Value-at-Risk."""
    var = value_at_risk(returns, confidence_level)
    tail_losses = returns[returns <= var]

    if len(tail_losses) == 0:
        return float(var)

    return float(tail_losses.mean())


def maximum_drawdown(returns):
    """Calculate maximum peak-to-trough decline."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def sharpe_ratio(returns, risk_free_rate=0.02):
    """Annualized Sharpe ratio using daily returns."""
    excess_daily_return = returns.mean() - (risk_free_rate / 252)

    if returns.std() == 0:
        return 0.0

    return float((excess_daily_return / returns.std()) * np.sqrt(252))


def market_risk_summary(portfolio_return_series):
    """Return key market-risk metrics for a portfolio."""
    return {
        "mean_daily_return": float(portfolio_return_series.mean()),
        "daily_volatility": float(portfolio_return_series.std()),
        "annualized_return": float(portfolio_return_series.mean() * 252),
        "annualized_volatility": float(portfolio_return_series.std() * np.sqrt(252)),
        "var_95": value_at_risk(portfolio_return_series, 0.95),
        "var_99": value_at_risk(portfolio_return_series, 0.99),
        "expected_shortfall_95": expected_shortfall(portfolio_return_series, 0.95),
        "expected_shortfall_99": expected_shortfall(portfolio_return_series, 0.99),
        "max_drawdown": maximum_drawdown(portfolio_return_series),
        "sharpe_ratio": sharpe_ratio(portfolio_return_series),
    }
