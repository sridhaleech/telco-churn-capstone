import argparse
from pathlib import Path

from src.analysis import explore, format_exploration
from src.data_loader import load_csv
from src.evaluation import format_metrics
from src.model import (
    build_predictions_frame,
    load_model,
    save_model,
    split_data,
    train_label_encoded_model,
    train_one_hot_model,
    train_random_forest,
)
from src.preprocessing import clean_dataframe, split_features_and_target
from src.validator import format_report, validate_dataframe


def run(input_path, output_path):
    print("TELCO CUSTOMER CHURN ANALYSIS")
    print("==============================")

    print("\n[1] Loading data...")
    raw = load_csv(input_path)
    print(f"Rows loaded: {len(raw)}")

    print("\n[2] Validating data...")
    print(format_report(validate_dataframe(raw)))

    print("\n[3] Cleaning data...")
    cleaned = clean_dataframe(raw)
    print("Cleaning complete. Raw CSV was not modified.")

    print("\n[4] Exploring data...")
    print(format_exploration(explore(cleaned)))

    print("\n[5] Training models...")
    features, target, customer_ids = split_features_and_target(cleaned)
    x_train, x_test, y_train, y_test, _training_ids, id_test = split_data(
        features, target, customer_ids
    )
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")

    label_metrics = train_label_encoded_model(x_train, x_test, y_train, y_test)
    one_hot_model, predictions, probabilities, one_hot_metrics = train_one_hot_model(
        x_train, x_test, y_train, y_test
    )
    forest_model, _forest_predictions, _forest_probabilities, forest_metrics = train_random_forest(
        x_train, x_test, y_train, y_test
    )

    print("\n[6] Evaluating models...")
    print(format_metrics("Logistic Regression with label encoding", label_metrics))
    print(format_metrics("Logistic Regression with one-hot encoding", one_hot_metrics))
    print(format_metrics("Random Forest", forest_metrics))
    print("The predictions file still uses one-hot Logistic Regression.")
    print("customerID was not used as a feature.")

    print("\n[7] Writing predictions...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame = build_predictions_frame(id_test, y_test, predictions, probabilities)
    predictions_frame.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(f"Prediction rows: {len(predictions_frame)}")

    model_path = output.parent / "churn_model.pkl"
    save_model(one_hot_model, model_path)
    print(f"Saved model: {model_path}")

    forest_path = output.parent / "random_forest_model.pkl"
    save_model(forest_model, forest_path)
    loaded_forest = load_model(forest_path)
    sample = x_test.iloc[[0]]
    sample_prediction = int(loaded_forest.predict(sample)[0])
    sample_probability = float(loaded_forest.predict_proba(sample)[0, 1])
    sample_actual = int(y_test.iloc[0])
    print(f"Saved Random Forest: {forest_path}")
    print("Reloaded the Random Forest from pickle and predicted one test customer:")
    print(f"customerID: {id_test.iloc[0]}")
    print(f"actual: {'Yes' if sample_actual == 1 else 'No'}")
    print(f"predicted: {'Yes' if sample_prediction == 1 else 'No'}")
    print(f"churn probability: {sample_probability:.3f}")
    print("\nPipeline completed successfully.")
    return one_hot_model, forest_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the telco churn analysis.")
    parser.add_argument("--input", default="data/raw/Telco-Customer-Churn.csv")
    parser.add_argument("--output", default="output/predictions.csv")
    args = parser.parse_args()
    run(args.input, args.output)
