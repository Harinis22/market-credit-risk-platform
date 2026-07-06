import numpy as np
import pandas as pd


def monte_carlo_portfolio_paths(
    current_value,
    mean_daily_return,
    daily_volatility,
    days=252,
    simulations=10000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    random_returns = rng.normal(mean_daily_return, daily_volatility, size=(days, simulations))
    paths = current_value * np.cumprod(1 + random_returns, axis=0)
    return pd.DataFrame(paths)


def monte_carlo_loss_summary(paths, current_value):
    final_values = paths.iloc[-1]
    pnl = final_values - current_value
    losses = -pnl

    return {
        "mean_final_value": float(final_values.mean()),
        "mean_pnl": float(pnl.mean()),
        "probability_of_loss": float((pnl < 0).mean()),
        "simulated_var_95": float(np.percentile(losses, 95)),
        "simulated_var_99": float(np.percentile(losses, 99)),
        "worst_1_percent_loss_avg": float(losses[losses >= np.percentile(losses, 99)].mean()),
    }


def expected_credit_loss(pd_values, lgd, ead_values):
    pd_values = np.asarray(pd_values, dtype=float)
    ead_values = np.asarray(ead_values, dtype=float)
    return float(np.sum(pd_values * lgd * ead_values))


def credit_stress_scenarios(base_pd, ead, lgd=0.45):
    scenarios = {
        "baseline": 1.00,
        "high_rate_environment": 1.25,
        "mild_recession": 1.50,
        "severe_recession": 2.00,
    }

    rows = []

    for name, multiplier in scenarios.items():
        stressed_pd = np.clip(np.asarray(base_pd) * multiplier, 0, 1)
        rows.append(
            {
                "scenario": name,
                "pd_multiplier": multiplier,
                "expected_credit_loss": expected_credit_loss(stressed_pd, lgd, ead),
                "avg_pd": float(np.mean(stressed_pd)),
            }
        )

    return pd.DataFrame(rows)
