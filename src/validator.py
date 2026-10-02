import pandas as pd

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

EXPECTED_CATEGORIES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_dataframe(df):
    """Check data quality. This function does not change the DataFrame."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    duplicate_ids = 0
    missing_fields = []
    invalid_target = 0
    total_charges_issues = 0
    negative_values = 0
    tenure_out_of_range = 0
    unexpected_categories = []

    if not missing_columns:
        duplicate_ids = int(df["customerID"].duplicated().sum())
        missing_fields = [name for name in df.columns if df[name].isna().sum() > 0]

        total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
        total_charges_issues = int(total_charges.isna().sum())
        negative_values = int((df["MonthlyCharges"] < 0).sum() + (total_charges.dropna() < 0).sum())
        tenure_out_of_range = int(((df["tenure"] < 0) | (df["tenure"] > 100)).sum())

        invalid_target = int((~df["Churn"].isin(["Yes", "No"])).sum())
        for column, allowed in EXPECTED_CATEGORIES.items():
            unexpected = sorted(set(df[column].dropna().unique()) - allowed)
            if unexpected:
                unexpected_categories.append(f"{column}: {unexpected}")

    needs_review = any([
        missing_columns,
        duplicate_ids,
        missing_fields,
        invalid_target,
        total_charges_issues,
        negative_values,
        tenure_out_of_range,
        unexpected_categories,
    ])

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_columns": missing_columns,
        "duplicate_ids": duplicate_ids,
        "missing_fields": missing_fields,
        "invalid_target": invalid_target,
        "total_charges_issues": total_charges_issues,
        "negative_values": negative_values,
        "tenure_out_of_range": tenure_out_of_range,
        "unexpected_categories": unexpected_categories,
        "status": "REVIEW" if needs_review else "PASS",
    }


def format_report(result):
    """Turn the validation result into text a non-programmer can read."""
    if result["missing_fields"]:
        missing_text = ", ".join(result["missing_fields"])
    else:
        missing_text = "none as NaN; TotalCharges has blank strings"

    if result["unexpected_categories"]:
        category_text = "; ".join(result["unexpected_categories"])
    else:
        category_text = "none"

    lines = [
        "DATA VALIDATION REPORT",
        "----------------------",
        f"Rows: {result['rows']}",
        f"Columns: {result['columns']}",
        f"Duplicate customer IDs: {result['duplicate_ids']}",
        f"Missing-value fields: {missing_text}",
        f"Invalid target values: {result['invalid_target']}",
        f"TotalCharges conversion issues: {result['total_charges_issues']}",
        f"Negative numeric values: {result['negative_values']}",
        f"Tenure outside 0-100 months: {result['tenure_out_of_range']}",
        f"Unexpected category values: {category_text}",
        f"Overall status: {result['status']}",
    ]
    return "\n".join(lines)
