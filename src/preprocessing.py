import pandas as pd


def clean_dataframe(df):
    """Return a cleaned copy. The original DataFrame and raw CSV stay unchanged."""
    cleaned = df.copy()

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    # Blank TotalCharges belong to new customers with tenure 0, so 0 is the honest fill.
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0)

    cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
    return cleaned


def split_features_and_target(cleaned):
    """Keep customerID for the output file, but do not use it as a model feature."""
    customer_ids = cleaned["customerID"]
    features = cleaned.drop(columns=["customerID", "Churn"])
    target = cleaned["Churn"]
    return features, target, customer_ids
