import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def assign_risk_band(probability_of_default):
    """Convert probability of default into a business-friendly risk band."""
    if probability_of_default < 0.10:
        return "Low Risk"
    if probability_of_default < 0.25:
        return "Medium Risk"
    if probability_of_default < 0.45:
        return "High Risk"
    return "Very High Risk"


def generate_synthetic_credit_data(rows=2500, seed=42):
    """
    Generate synthetic borrower-level credit-risk data.

    This is used only for demonstration because real borrower data is private.
    The variables are clipped to realistic ranges to avoid invalid outliers.
    """
    rng = np.random.default_rng(seed)

    annual_income = rng.normal(85000, 30000, rows).clip(25000, 220000)
    loan_amount = rng.normal(25000, 12000, rows).clip(2000, 75000)
    interest_rate = rng.normal(0.13, 0.055, rows).clip(0.035, 0.32)
    debt_to_income = rng.normal(0.32, 0.16, rows).clip(0.02, 0.75)
    delinquencies = rng.poisson(0.65, rows).clip(0, 6)
    credit_history_years = rng.normal(9, 5, rows).clip(1, 35)
    utilization = rng.normal(0.45, 0.22, rows).clip(0.02, 0.99)

    # Default risk formula used to create a realistic synthetic target.
    risk_score = (
        -3.2
        + 2.8 * debt_to_income
        + 2.2 * utilization
        + 1.8 * interest_rate
        + 0.45 * delinquencies
        - 0.000012 * annual_income
        - 0.045 * credit_history_years
        + 0.00001 * loan_amount
    )

    probability_of_default = 1 / (1 + np.exp(-risk_score))
    default = rng.binomial(1, probability_of_default)

    df = pd.DataFrame(
        {
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "interest_rate": interest_rate.round(4),
            "debt_to_income": debt_to_income.round(4),
            "delinquencies": delinquencies.astype(int),
            "credit_history_years": credit_history_years.round(1),
            "utilization": utilization.round(4),
            "default": default.astype(int),
        }
    )

    return df


def train_credit_models(df):
    """
    Train Logistic Regression and Random Forest models for credit default prediction.

    Returns:
        results: model outputs and evaluation metrics
        comparison_df: model comparison table
        best_model_name: model with highest ROC-AUC
        best_result: result dictionary for the best model
    """
    features = [
        "annual_income",
        "loan_amount",
        "interest_rate",
        "debt_to_income",
        "delinquencies",
        "credit_history_years",
        "utilization",
    ]

    X = df[features]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest,
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        default_probability = model.predict_proba(X_test)[:, 1]
        predictions = (default_probability >= 0.50).astype(int)

        results[name] = {
            "model": model,
            "roc_auc": float(roc_auc_score(y_test, default_probability)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
            "test_pd": default_probability,
            "test_ead": X_test["loan_amount"].to_numpy(),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "features": features,
        }

    comparison_df = pd.DataFrame(
        [
            {
                "Model": name,
                "ROC-AUC": result["roc_auc"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1_score"],
            }
            for name, result in results.items()
        ]
    ).sort_values("ROC-AUC", ascending=False)

    best_model_name = comparison_df.iloc[0]["Model"]
    best_result = results[best_model_name]

    return results, comparison_df, best_model_name, best_result
