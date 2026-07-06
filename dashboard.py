import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.credit_risk import (
    assign_risk_band,
    generate_synthetic_credit_data,
    train_credit_models,
)
from src.market_risk import (
    calculate_returns,
    market_risk_summary,
    portfolio_returns,
)
from src.simulation import (
    credit_stress_scenarios,
    monte_carlo_loss_summary,
    monte_carlo_portfolio_paths,
)


st.set_page_config(
    page_title="Market & Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
)


def generate_synthetic_prices(days=756, seed=42):
    """Create synthetic daily prices for a diversified portfolio."""
    rng = np.random.default_rng(seed)

    tickers = ["SPY", "QQQ", "IWM", "JPM", "MSFT"]

    assumptions = {
        "SPY": (0.00035, 0.011),
        "QQQ": (0.00045, 0.014),
        "IWM": (0.00030, 0.016),
        "JPM": (0.00032, 0.015),
        "MSFT": (0.00050, 0.014),
    }

    prices = pd.DataFrame(
        index=pd.date_range(
            end=pd.Timestamp.today(),
            periods=days,
            freq="B",
        )
    )

    for ticker in tickers:
        mean_return, volatility = assumptions[ticker]
        daily_returns = rng.normal(mean_return, volatility, days)
        prices[ticker] = 100 * np.cumprod(1 + daily_returns)

    return prices


@st.cache_data
def load_project_outputs(days, portfolio_value, simulations):
    """Run project calculations once and cache dashboard outputs."""
    prices = generate_synthetic_prices(days=days)
    returns = calculate_returns(prices)

    weights = np.array([0.35, 0.20, 0.15, 0.15, 0.15])
    port_returns = portfolio_returns(returns, weights)

    risk_summary = market_risk_summary(port_returns)

    paths = monte_carlo_portfolio_paths(
        current_value=portfolio_value,
        mean_daily_return=port_returns.mean(),
        daily_volatility=port_returns.std(),
        days=252,
        simulations=simulations,
    )

    mc_summary = monte_carlo_loss_summary(paths, portfolio_value)

    credit_df = generate_synthetic_credit_data(rows=2500)
    model_results = train_credit_models(credit_df)

    best_model_name = max(model_results, key=lambda name: model_results[name]["roc_auc"])
    best_result = model_results[best_model_name]

    ecl_summary = credit_stress_scenarios(
        base_pd=best_result["test_pd"],
        ead=best_result["test_ead"],
        lgd=0.45,
    )

    return {
        "prices": prices,
        "returns": returns,
        "portfolio_returns": port_returns,
        "risk_summary": risk_summary,
        "paths": paths,
        "mc_summary": mc_summary,
        "credit_df": credit_df,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_result": best_result,
        "ecl_summary": ecl_summary,
    }


st.title("Integrated Market & Credit Risk Simulation Platform")

st.caption(
    "Interactive Python dashboard combining market risk metrics, Monte Carlo simulation, "
    "credit default prediction, and expected credit loss stress testing."
)

st.sidebar.header("Dashboard Controls")

portfolio_value = st.sidebar.number_input(
    "Portfolio Value",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
)

days = st.sidebar.slider(
    "Historical Price Days",
    min_value=252,
    max_value=1260,
    value=756,
    step=126,
)

simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=1000,
)

outputs = load_project_outputs(days, portfolio_value, simulations)

risk_summary = outputs["risk_summary"]
mc_summary = outputs["mc_summary"]
paths = outputs["paths"]
port_returns = outputs["portfolio_returns"]
prices = outputs["prices"]
credit_df = outputs["credit_df"]
best_result = outputs["best_result"]
best_model_name = outputs["best_model_name"]
ecl_summary = outputs["ecl_summary"]

avg_pd = float(np.mean(best_result["test_pd"]))

st.subheader("Executive Risk Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Annualized Volatility",
    f"{risk_summary['annualized_volatility']:.2%}",
)

col2.metric(
    "95% VaR",
    f"{risk_summary['var_95']:.2%}",
)

col3.metric(
    "99% VaR",
    f"{risk_summary['var_99']:.2%}",
)

col4.metric(
    "Model ROC-AUC",
    f"{best_result['roc_auc']:.3f}",
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Sharpe Ratio",
    f"{risk_summary['sharpe_ratio']:.2f}",
)

col6.metric(
    "Max Drawdown",
    f"{risk_summary['max_drawdown']:.2%}",
)

col7.metric(
    "Probability of Loss",
    f"{mc_summary['probability_of_loss']:.2%}",
)

col8.metric(
    "Avg Predicted PD",
    f"{avg_pd:.2%}",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Market Risk",
        "Monte Carlo Simulation",
        "Credit Risk ML",
        "Stress Testing",
        "Interview Summary",
    ]
)

with tab1:
    st.subheader("Market Risk Analysis")

    cumulative_returns = (1 + port_returns).cumprod()

    fig = px.line(
        cumulative_returns,
        title="Portfolio Cumulative Return",
        labels={"index": "Date", "value": "Growth of $1"},
    )
    st.plotly_chart(fig, use_container_width=True)

    risk_df = pd.DataFrame(
        {
            "Metric": list(risk_summary.keys()),
            "Value": list(risk_summary.values()),
        }
    )

    st.write("Market Risk Metrics")
    st.dataframe(risk_df, use_container_width=True)

    returns_fig = px.histogram(
        port_returns,
        nbins=60,
        title="Portfolio Daily Return Distribution",
        labels={"value": "Daily Return"},
    )
    st.plotly_chart(returns_fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Portfolio Simulation")

    display_paths = paths.iloc[:, :100]

    fig = go.Figure()

    for col in display_paths.columns:
        fig.add_trace(
            go.Scatter(
                y=display_paths[col],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Monte Carlo Simulation - First 100 Portfolio Paths",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value",
    )

    st.plotly_chart(fig, use_container_width=True)

    mc_df = pd.DataFrame(
        {
            "Metric": list(mc_summary.keys()),
            "Value": list(mc_summary.values()),
        }
    )

    st.write("Monte Carlo Loss Summary")
    st.dataframe(mc_df, use_container_width=True)

    final_values = paths.iloc[-1]
    final_fig = px.histogram(
        final_values,
        nbins=60,
        title="Distribution of Simulated Final Portfolio Values",
        labels={"value": "Final Portfolio Value"},
    )
    st.plotly_chart(final_fig, use_container_width=True)

with tab3:
    st.subheader("Credit Risk Machine Learning")

    st.write(f"Best Model: **{best_model_name}**")
    st.write(f"ROC-AUC: **{best_result['roc_auc']:.4f}**")
    st.write(f"Average Probability of Default: **{avg_pd:.2%}**")
    st.write(f"Average Risk Band: **{assign_risk_band(avg_pd)}**")

    credit_preview = credit_df.head(20)
    st.write("Sample Borrower Data")
    st.dataframe(credit_preview, use_container_width=True)

    pd_values = pd.Series(best_result["test_pd"], name="Probability of Default")
    pd_fig = px.histogram(
        pd_values,
        nbins=50,
        title="Predicted Probability of Default Distribution",
    )
    st.plotly_chart(pd_fig, use_container_width=True)

    cm = np.array(best_result["confusion_matrix"])

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-Default", "Default"],
        y=["Non-Default", "Default"],
    )
    st.plotly_chart(cm_fig, use_container_width=True)

with tab4:
    st.subheader("Expected Credit Loss Stress Testing")

    ecl_fig = px.bar(
        ecl_summary,
        x="scenario",
        y="expected_credit_loss",
        title="Expected Credit Loss by Scenario",
        labels={
            "scenario": "Scenario",
            "expected_credit_loss": "Expected Credit Loss",
        },
    )
    st.plotly_chart(ecl_fig, use_container_width=True)

    st.write("Credit Stress Scenario Table")
    st.dataframe(ecl_summary, use_container_width=True)

    st.info(
        "Expected Credit Loss is calculated using PD × LGD × EAD. "
        "The stress scenarios increase default probability to estimate how losses may rise "
        "during high-rate or recession environments."
    )

with tab5:
    st.subheader("Resume and Interview Explanation")

    st.markdown(
        """
        ### Resume Bullet

        Built an integrated **Market & Credit Risk Simulation Platform** using Python, Pandas, NumPy,
        Scikit-learn, and Monte Carlo simulation to estimate VaR, Expected Shortfall, portfolio drawdown risk,
        borrower default probability, and expected credit loss under baseline, high-rate, and recession stress scenarios.

        ### Interview Explanation

        This project demonstrates a combined market risk and credit risk workflow.

        For **market risk**, the dashboard calculates daily returns, volatility, Sharpe ratio,
        maximum drawdown, VaR, and Expected Shortfall for a synthetic multi-asset portfolio.

        For **simulation**, it applies Monte Carlo methods to generate thousands of future
        portfolio paths and estimate downside loss exposure.

        For **credit risk**, it generates borrower-level data and trains Logistic Regression
        and Random Forest models to predict Probability of Default.

        For **stress testing**, it uses PD, LGD, and EAD to calculate Expected Credit Loss
        under baseline, high-rate, mild recession, and severe recession scenarios.
        """
    )
