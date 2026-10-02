from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(actual, predicted):
    """Score a binary churn model. The positive class is 1, meaning churned."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted),
        "classification_report": classification_report(
            actual,
            predicted,
            target_names=["Stay (0)", "Churn (1)"],
            zero_division=0,
        ),
    }


def format_metrics(name, metrics):
    matrix = metrics["confusion_matrix"]
    lines = [
        f"MODEL: {name}",
        f"Accuracy : {metrics['accuracy']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall   : {metrics['recall']:.3f}",
        f"F1 score : {metrics['f1']:.3f}",
        "Confusion matrix rows are actual Stay/Churn, columns are predicted Stay/Churn:",
        str(matrix),
        metrics["classification_report"],
    ]
    return "\n".join(lines)
