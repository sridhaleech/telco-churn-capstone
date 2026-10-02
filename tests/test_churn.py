import pandas as pd

from src.model import build_predictions_frame, split_data, train_one_hot_model
from src.preprocessing import clean_dataframe
from src.validator import REQUIRED_COLUMNS, validate_dataframe


def sample_row(**overrides):
    row = {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "250",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_missing_required_column_is_reported():
    frame = pd.DataFrame([sample_row()]).drop(columns=["Churn"])
    result = validate_dataframe(frame)
    assert "Churn" in result["missing_columns"]
    assert result["status"] == "REVIEW"


def test_invalid_churn_value_is_detected():
    frame = pd.DataFrame([sample_row(Churn="Maybe")])
    result = validate_dataframe(frame)
    assert result["invalid_target"] == 1


def test_duplicate_customer_id_is_detected():
    frame = pd.DataFrame([
        sample_row(customerID="1111-AAAA"),
        sample_row(customerID="1111-AAAA", Churn="Yes"),
    ])
    result = validate_dataframe(frame)
    assert result["duplicate_ids"] == 1


def test_blank_total_charges_become_zero():
    frame = pd.DataFrame([sample_row(tenure=0, TotalCharges=" ")])
    cleaned = clean_dataframe(frame)
    assert cleaned.loc[0, "TotalCharges"] == 0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_cleaned_charges_are_not_negative():
    frame = pd.DataFrame([
        sample_row(MonthlyCharges=29.85, TotalCharges="29.85"),
        sample_row(customerID="2222-BBBB", tenure=0, TotalCharges=" ", Churn="Yes"),
    ])
    cleaned = clean_dataframe(frame)
    assert (cleaned["MonthlyCharges"] >= 0).all()
    assert (cleaned["TotalCharges"] >= 0).all()


def test_prediction_output_has_required_columns():
    frame = build_predictions_frame(
        customer_ids=pd.Series(["7590-VHVEG"]),
        actual=pd.Series([1]),
        predicted=[0],
        probabilities=[0.42],
    )
    assert list(frame.columns) == [
        "customerID",
        "actual_churn",
        "predicted_churn",
        "churn_probability",
    ]
    assert set(frame["predicted_churn"]).issubset({"Yes", "No"})


def test_model_predictions_use_only_valid_classes():
    rows = []
    for index in range(20):
        churn = "Yes" if index < 6 else "No"
        rows.append(sample_row(customerID=f"CUST-{index:04d}", tenure=index, Churn=churn))
    cleaned = clean_dataframe(pd.DataFrame(rows))
    features = cleaned.drop(columns=["customerID", "Churn"])
    target = cleaned["Churn"]
    ids = cleaned["customerID"]
    x_train, x_test, y_train, y_test, _ids_train, _ids_test = split_data(features, target, ids)
    _model, predictions, _probabilities, _metrics = train_one_hot_model(
        x_train, x_test, y_train, y_test
    )
    assert set(predictions).issubset({0, 1})
    assert set(REQUIRED_COLUMNS) == set(sample_row())
