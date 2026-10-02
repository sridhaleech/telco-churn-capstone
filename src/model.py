import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation import calculate_metrics


def numeric_and_categorical_columns(features):
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def split_data(features, target, customer_ids):
    """Hold out 30% for testing, as requested in class. The split stays the same every run."""
    return train_test_split(
        features,
        target,
        customer_ids,
        test_size=0.30,
        random_state=42,
        stratify=target,
    )


def _label_encode(train_df, test_df, columns):
    """Turn each category into a number using only categories seen in training."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    for column in columns:
        categories = {
            value: index for index, value in enumerate(sorted(train_df[column].astype(str).unique()))
        }
        train_encoded[column] = train_df[column].astype(str).map(categories)
        test_encoded[column] = test_df[column].astype(str).map(categories).fillna(-1)
    return train_encoded, test_encoded


def train_label_encoded_model(x_train, x_test, y_train, y_test):
    numeric, categorical = numeric_and_categorical_columns(x_train)
    train_encoded, test_encoded = _label_encode(x_train, x_test, categorical)

    scaler = StandardScaler()
    train_encoded[numeric] = scaler.fit_transform(train_encoded[numeric])
    test_encoded[numeric] = scaler.transform(test_encoded[numeric])

    model = LogisticRegression(max_iter=1000)
    model.fit(train_encoded, y_train)
    predictions = model.predict(test_encoded)
    return calculate_metrics(y_test, predictions)


def train_one_hot_model(x_train, x_test, y_train, y_test):
    """One-hot encoding gives each category its own 0/1 column. This is the preferred model."""
    numeric, categorical = numeric_and_categorical_columns(x_train)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions)
    return model, predictions, probabilities, metrics


def save_model(model, path):
    """Save the trained one-hot model so it can be loaded again without retraining."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)


def load_model(path):
    """Load a model that was saved with pickle."""
    with Path(path).open("rb") as model_file:
        return pickle.load(model_file)


def build_predictions_frame(customer_ids, actual, predicted, probabilities):
    return pd.DataFrame(
        {
            "customerID": customer_ids.to_numpy(),
            "actual_churn": pd.Series(actual).map({1: "Yes", 0: "No"}).to_numpy(),
            "predicted_churn": pd.Series(predicted).map({1: "Yes", 0: "No"}).to_numpy(),
            "churn_probability": probabilities,
        }
    )
