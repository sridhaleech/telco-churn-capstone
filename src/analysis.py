import pandas as pd


def churn_rate_by(df, column):
    """Return the churn rate for each value in a column, highest first."""
    summary = (
        df.groupby(column, observed=True)["Churn"]
        .agg(customers="size", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
    )
    summary["churn_rate"] = (summary["churn_rate"] * 100).round(1)
    return summary


def explore(cleaned):
    """Calculate the churn rate and the segment comparisons the coach requested."""
    explored = cleaned.copy()
    explored["tenure_group"] = pd.cut(
        explored["tenure"],
        bins=[-1, 12, 24, 48, 100],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-100 months"],
    )

    overall_rate = round(explored["Churn"].mean() * 100, 1)
    by_contract = churn_rate_by(explored, "Contract")
    by_internet = churn_rate_by(explored, "InternetService")
    by_tenure = churn_rate_by(explored, "tenure_group")
    by_payment = churn_rate_by(explored, "PaymentMethod")
    charges = explored.groupby("Churn")["MonthlyCharges"].mean().round(2)

    report = {
        "overall_churn_percent": overall_rate,
        "by_contract": by_contract,
        "by_internet": by_internet,
        "by_tenure": by_tenure,
        "by_payment": by_payment,
        "average_monthly_charges": charges,
    }
    return report


def format_exploration(report):
    """Turn the calculations into sentences that can be explained in class."""
    charges = report["average_monthly_charges"]
    lines = [
        "EXPLORATION FINDINGS",
        "--------------------",
        f"1. About {report['overall_churn_percent']}% of customers churned.",
        "2. Churn by contract:",
        report["by_contract"].to_string(),
        "3. Churn by internet service:",
        report["by_internet"].to_string(),
        "4. Churn by tenure:",
        report["by_tenure"].to_string(),
        "5. Churn by payment method:",
        report["by_payment"].to_string(),
        (
            f"6. Average monthly charges are {charges.get(1)} for churned customers "
            f"and {charges.get(0)} for customers who stayed."
        ),
        "These are patterns in the snapshot. They do not prove that a change would stop churn.",
    ]
    return "\n".join(lines)
