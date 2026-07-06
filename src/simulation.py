import numpy as np
import pandas as pd


def monte_carlo_portfolio_paths(
    current_value=100000,
    mean_daily_return=0.0003,
    daily_volatility=0.012,
    days=252,
    simulations=5000,
    seed=42,
):
    """Simulate future portfolio paths using daily normal return assumptions."""
    rng = np.random.default_rng(seed)

    simulated_returns = rng.normal(
        loc=mean_daily_return,
        scale=daily_volatility,
        size=(days, simulations),
    )

    paths = current_value * np.cumprod(1 + simulated_returns, axis=0)

    return pd.DataFrame(paths)


def monte_carlo_loss_summary(paths, starting_value):
    """Summarize downside loss from Monte Carlo simulated paths."""
    final_values = paths.iloc[-1]
    losses = starting_value - final_values

    return {
        "mean_final_value": float(final_values.mean()),
        "median_final_value": float(final_values.median()),
        "fifth_percentile_final_value": float(final_values.quantile(0.05)),
        "first_percentile_final_value": float(final_values.quantile(0.01)),
        "mean_loss": float(losses.mean()),
        "var_95_loss": float(losses.quantile(0.95)),
        "var_99_loss": float(losses.quantile(0.99)),
        "probability_of_loss": float((final_values < starting_value).mean()),
    }


def expected_credit_loss(pd_values, lgd, ead):
    """Calculate Expected Credit Loss using ECL = PD × LGD × EAD."""
    pd_values = np.array(pd_values, dtype=float)
    ead = np.array(ead, dtype=float)

    return pd_values * lgd * ead


def credit_stress_scenarios(base_pd, ead, lgd=0.45):
    """
    Calculate expected credit loss under multiple stress scenarios.

    base_pd: model-predicted probability of default
    ead: exposure at default, represented by loan amount in this project
    lgd: loss given default assumption
    """
    scenarios = {
        "baseline": 1.00,
        "high_rate_environment": 1.25,
        "mild_recession": 1.60,
        "severe_recession": 2.10,
    }

    rows = []

    for scenario, multiplier in scenarios.items():
        stressed_pd = np.clip(np.array(base_pd) * multiplier, 0, 1)
        ecl = expected_credit_loss(stressed_pd, lgd, ead)

        rows.append(
            {
                "scenario": scenario,
                "pd_multiplier": multiplier,
                "average_pd": float(stressed_pd.mean()),
                "expected_credit_loss": float(ecl.sum()),
                "average_ecl_per_borrower": float(ecl.mean()),
            }
        )

    return pd.DataFrame(rows)
