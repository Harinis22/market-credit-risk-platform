import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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
    """Create synthetic daily prices for a diversified multi-asset portfolio."""
    rng = np.random.default_rng(seed)

    tickers = ["SPY", "QQQ", "IWM", "JPM", "MSFT", "GLD", "TLT"]

    assumptions = {
        "SPY": (0.00035, 0.011),
        "QQQ": (0.00045, 0.014),
        "IWM": (0.00030, 0.016),
        "JPM": (0.00032, 0.015),
        "MSFT": (0.00050, 0.014),
        "GLD": (0.00020, 0.010),
        "TLT": (0.00015, 0.012),
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


def load_yfinance_prices(tickers, period="5y", interval="1d"):
    """Download real historical adjusted closing prices from yfinance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Add yfinance>=0.2.40 to requirements.txt."
        )

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data returned from yfinance. Try fewer tickers or another period.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise ValueError("Could not find Close prices in yfinance response.")
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()

    if prices.empty:
        raise ValueError("Price data is empty after cleaning.")

    return prices


def get_market_prices(data_source, days, tickers, period):
    """Return market price data from either synthetic generator or yfinance."""
    if data_source == "Real market data from yfinance":
        return load_yfinance_prices(tickers=tickers, period=period, interval="1d")

    return generate_synthetic_prices(days=days)


def align_weights_to_prices(prices):
    """Create default portfolio weights aligned to the available ticker columns."""
    default_weights = {
        "SPY": 0.30,
        "QQQ": 0.20,
        "IWM": 0.10,
        "JPM": 0.10,
        "MSFT": 0.10,
        "GLD": 0.10,
        "TLT": 0.10,
    }

    columns = list(prices.columns)

    raw_weights = np.array(
        [default_weights.get(ticker, 1.0 / len(columns)) for ticker in columns],
        dtype=float,
    )

    weights = raw_weights / raw_weights.sum()
    return weights


@st.cache_data(show_spinner=True)
def load_project_outputs(data_source, days, portfolio_value, simulations, tickers, period):
    """Run project calculations and cache dashboard outputs."""
    prices = get_market_prices(data_source, days, tickers, period)

    returns = calculate_returns(prices)
    weights = align_weights_to_prices(prices)
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
    model_results, comparison_df, best_model_name, best_result = train_credit_models(credit_df)

    ecl_summary = credit_stress_scenarios(
        base_pd=best_result["test_pd"],
        ead=best_result["test_ead"],
        lgd=0.45,
    )

    return {
        "prices": prices,
        "weights": weights,
        "returns": returns,
        "portfolio_returns": port_returns,
        "risk_summary": risk_summary,
        "paths": paths,
        "mc_summary": mc_summary,
        "credit_df": credit_df,
        "model_results": model_results,
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_result": best_result,
        "ecl_summary": ecl_summary,
    }


st.title("Integrated Market & Credit Risk Simulation Platform")

st.caption(
    "Interactive Python dashboard combining real yfinance market data, market risk metrics, "
    "Monte Carlo simulation, credit default prediction, model comparison, SHAP explainability, "
    "and expected credit loss stress testing."
)

st.sidebar.header("Dashboard Controls")

data_source = st.sidebar.radio(
    "Market Data Source",
    ["Synthetic demo data", "Real market data from yfinance"],
    index=1,
)

default_tickers = "SPY,QQQ,IWM,JPM,MSFT,GLD,TLT"

ticker_text = st.sidebar.text_input(
    "Ticker List",
    value=default_tickers,
    help="Used only when yfinance mode is selected.",
)

tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]

period = st.sidebar.selectbox(
    "yfinance Historical Period",
    ["1y", "2y", "5y", "10y"],
    index=2,
)

portfolio_value = st.sidebar.number_input(
    "Portfolio Value",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
)

days = st.sidebar.slider(
    "Synthetic Historical Price Days",
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

if data_source == "Real market data from yfinance" and not YFINANCE_AVAILABLE:
    st.error(
        "yfinance is not installed. Add `yfinance>=0.2.40` to requirements.txt, "
        "then run `python -m pip install -r requirements.txt`."
    )
    st.stop()

try:
    outputs = load_project_outputs(
        data_source=data_source,
        days=days,
        portfolio_value=portfolio_value,
        simulations=simulations,
        tickers=tickers,
        period=period,
    )
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.info("Switch to synthetic demo data or check your ticker symbols and internet connection.")
    st.stop()

risk_summary = outputs["risk_summary"]
mc_summary = outputs["mc_summary"]
paths = outputs["paths"]
port_returns = outputs["portfolio_returns"]
prices = outputs["prices"]
weights = outputs["weights"]
credit_df = outputs["credit_df"]
best_result = outputs["best_result"]
best_model_name = outputs["best_model_name"]
ecl_summary = outputs["ecl_summary"]
comparison_df = outputs["comparison_df"]

avg_pd = float(np.mean(best_result["test_pd"]))

st.subheader("Executive Risk Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annualized Volatility", f"{risk_summary['annualized_volatility']:.2%}")
col2.metric("95% VaR", f"{risk_summary['var_95']:.2%}")
col3.metric("99% VaR", f"{risk_summary['var_99']:.2%}")
col4.metric("Best Model ROC-AUC", f"{best_result['roc_auc']:.3f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Sharpe Ratio", f"{risk_summary['sharpe_ratio']:.2f}")
col6.metric("Max Drawdown", f"{risk_summary['max_drawdown']:.2%}")
col7.metric("Probability of Loss", f"{mc_summary['probability_of_loss']:.2%}")
col8.metric("Avg Predicted PD", f"{avg_pd:.2%}")

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

    st.write(f"Market Data Source: **{data_source}**")

    if data_source == "Real market data from yfinance":
        st.success(
            "Using real historical adjusted closing prices from yfinance for the market-risk module."
        )
    else:
        st.info(
            "Using synthetic market data for stable demos. Switch to yfinance mode for real historical prices."
        )

    st.write("Portfolio Weights")
    weight_df = pd.DataFrame(
        {
            "Ticker": list(prices.columns),
            "Weight": weights,
        }
    )
    st.dataframe(weight_df, use_container_width=True)

    st.write("Recent Price Data")
    st.dataframe(prices.tail(10), use_container_width=True)

    normalized_prices = prices / prices.iloc[0]

    price_fig = px.line(
        normalized_prices,
        title="Normalized Price Performance",
        labels={"value": "Growth of $1", "index": "Date"},
    )
    st.plotly_chart(price_fig, use_container_width=True)

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

    correlation_fig = px.imshow(
        outputs["returns"].corr(),
        text_auto=True,
        title="Asset Return Correlation Matrix",
        labels=dict(color="Correlation"),
    )
    st.plotly_chart(correlation_fig, use_container_width=True)

    st.info(
        "Observation: most returns cluster around the average, while the far-left tail represents downside-risk observations. "
        "Those tail losses drive VaR, Expected Shortfall, and stress-risk analysis."
    )

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

    st.info(
        "Observation: most simulated paths remain in a normal range, while a smaller number of paths show extreme downside outcomes. "
        "These are valid tail-risk scenarios, not data errors."
    )

with tab3:
    st.subheader("Credit Risk Machine Learning")

    st.write(f"Best Model Based on ROC-AUC: **{best_model_name}**")
    st.write(f"Best Model ROC-AUC: **{best_result['roc_auc']:.4f}**")
    st.write(f"Average Probability of Default: **{avg_pd:.2%}**")
    st.write(f"Average Risk Band: **{assign_risk_band(avg_pd)}**")

    st.write("Model Comparison: Logistic Regression vs Random Forest")
    st.dataframe(comparison_df, use_container_width=True)

    comparison_fig = px.bar(
        comparison_df,
        x="Model",
        y=["ROC-AUC", "Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group",
        title="Credit Risk Model Performance Comparison",
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

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
        title="Confusion Matrix for Best Credit Risk Model",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-Default", "Default"],
        y=["Non-Default", "Default"],
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.subheader("SHAP Explanation for Credit Risk Model")

    st.write(
        "SHAP explainability helps identify which borrower features push predictions toward higher or lower default risk."
    )

    if SHAP_AVAILABLE:
        rf_result = outputs["model_results"]["Random Forest"]
        rf_model = rf_result["model"]
        X_test = rf_result["X_test"]
        features = rf_result["features"]

        sample_size = min(200, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)

        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values_for_default = shap_values[1]
            elif len(np.array(shap_values).shape) == 3:
                shap_values_for_default = np.array(shap_values)[:, :, 1]
            else:
                shap_values_for_default = shap_values

            shap_importance = pd.DataFrame(
                {
                    "Feature": features,
                    "Mean Absolute SHAP Value": np.abs(shap_values_for_default).mean(axis=0),
                }
            ).sort_values("Mean Absolute SHAP Value", ascending=False)

            st.write("Top Features Driving Default Risk")
            st.dataframe(shap_importance, use_container_width=True)

            shap_fig = px.bar(
                shap_importance,
                x="Mean Absolute SHAP Value",
                y="Feature",
                orientation="h",
                title="SHAP Feature Importance for Default Prediction",
            )
            shap_fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(shap_fig, use_container_width=True)

            st.info(
                "Interpretation: larger mean absolute SHAP values indicate features that have stronger influence on default-risk predictions. "
                "Typical drivers include debt-to-income ratio, utilization, interest rate, delinquencies, income, and credit history."
            )

        except Exception as exc:
            st.warning(
                f"SHAP could not be calculated in this environment. Error: {exc}. "
                "The model comparison table and confusion matrix are still available."
            )
    else:
        st.warning(
            "SHAP is not installed. Add `shap>=0.45` to requirements.txt, push to GitHub, and reboot Streamlit Cloud."
        )

    st.info(
        "Credit-risk observation: high DTI, high utilization, higher interest rates, and multiple delinquencies create high-risk borrower profiles. "
        "These are valid risk outliers rather than invalid data points."
    )

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
        "The stress scenarios increase default probability to estimate how credit losses may rise "
        "during high-rate or recession environments."
    )

    scenario_notes = pd.DataFrame(
        {
            "Scenario": ["baseline", "high_rate_environment", "mild_recession", "severe_recession"],
            "Business Interpretation": [
                "Normal market and credit environment.",
                "Borrowers face higher interest-rate pressure.",
                "Default probabilities rise due to weaker economic conditions.",
                "Severe stress combines elevated default risk and larger expected losses.",
            ],
        }
    )

    st.write("Scenario Interpretation")
    st.dataframe(scenario_notes, use_container_width=True)

with tab5:
    st.subheader("Resume and Interview Explanation")

    st.markdown(
        """
        ### Resume Bullet

        Built an integrated **Market & Credit Risk Simulation Platform** using Python, Pandas, NumPy,
        Scikit-learn, yfinance, Monte Carlo simulation, Streamlit, and SHAP explainability to estimate VaR,
        Expected Shortfall, portfolio drawdown risk, borrower default probability, model performance,
        feature-level risk drivers, and expected credit loss under baseline, high-rate, and recession stress scenarios.

        ### Interview Explanation

        This project demonstrates a combined market risk and credit risk workflow.

        For **market risk**, the dashboard can use real historical adjusted closing prices from yfinance
        for SPY, QQQ, IWM, JPM, MSFT, GLD, and TLT. It calculates daily returns, volatility,
        Sharpe ratio, maximum drawdown, VaR, Expected Shortfall, and asset correlations.

        For **simulation**, it applies Monte Carlo methods to generate thousands of future
        portfolio paths and estimate downside loss exposure.

        For **credit risk**, it generates borrower-level data and trains Logistic Regression
        and Random Forest models to predict Probability of Default.

        For **model comparison**, it compares Logistic Regression and Random Forest using
        ROC-AUC, accuracy, precision, recall, and F1-score.

        For **explainability**, it uses SHAP feature importance to explain which borrower
        variables contribute most to predicted default risk.

        For **stress testing**, it uses PD, LGD, and EAD to calculate Expected Credit Loss
        under baseline, high-rate, mild recession, and severe recession scenarios.

        ### Important Interview Note

        The market-risk section can use real yfinance market data. The credit-risk section uses
        synthetic borrower data to demonstrate the workflow safely without exposing private loan records.
        """
    )
