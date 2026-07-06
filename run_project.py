import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.credit_risk import generate_synthetic_credit_data, train_credit_models
from src.market_risk import calculate_returns, market_risk_summary, portfolio_returns
from src.simulation import (
    credit_stress_scenarios,
    monte_carlo_loss_summary,
    monte_carlo_portfolio_paths,
)


def load_yfinance_prices(tickers, period="5y", interval="1d"):
    """Download real historical adjusted closing prices from yfinance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("Install yfinance first: python -m pip install yfinance")

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    return prices.dropna(how="all").ffill().dropna()


def main():
    tickers = ["SPY", "QQQ", "IWM", "JPM", "MSFT", "GLD", "TLT"]
    weights = np.array([0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10])

    prices = load_yfinance_prices(tickers=tickers, period="5y")
    returns = calculate_returns(prices)
    port_returns = portfolio_returns(returns, weights)

    risk_summary = market_risk_summary(port_returns)

    print("\nMARKET RISK SUMMARY")
    for metric, value in risk_summary.items():
        print(f"{metric}: {value}")

    paths = monte_carlo_portfolio_paths(
        current_value=100000,
        mean_daily_return=port_returns.mean(),
        daily_volatility=port_returns.std(),
        days=252,
        simulations=5000,
    )

    mc_summary = monte_carlo_loss_summary(paths, starting_value=100000)

    print("\nMONTE CARLO LOSS SUMMARY")
    for metric, value in mc_summary.items():
        print(f"{metric}: {value}")

    credit_df = generate_synthetic_credit_data(rows=2500)
    model_results, comparison_df, best_model_name, best_result = train_credit_models(credit_df)

    print("\nCREDIT MODEL COMPARISON")
    print(comparison_df)

    print(f"\nBEST CREDIT MODEL: {best_model_name}")
    print(f"ROC-AUC: {best_result['roc_auc']:.4f}")

    ecl_summary = credit_stress_scenarios(
        base_pd=best_result["test_pd"],
        ead=best_result["test_ead"],
        lgd=0.45,
    )

    print("\nEXPECTED CREDIT LOSS SCENARIOS")
    print(ecl_summary)


if __name__ == "__main__":
    main()
