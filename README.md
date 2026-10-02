# Telco Customer Churn Prediction

A small Python program that loads telecom customer data, checks its quality, cleans a copy, explores churn patterns, and trains a Logistic Regression model.

This is an educational classification exercise. The results show patterns. They do not prove that a change, such as moving a customer to a two-year contract, will stop churn.

## Business problem

A telecom company wants to know which customers are more likely to leave. The question is:

Can tenure, contract type, services, payment method, monthly charges, and the other customer attributes predict whether a customer will churn?

## Dataset

Source: IBM Telco Customer Churn sample, file `Telco-Customer-Churn.csv`.

https://github.com/IBM/telco-customer-churn-on-icp4d

| Item | Value |
| --- | --- |
| Rows | 7,043 customers |
| Raw columns | 21 |
| Target | `Churn` (`Yes` or `No`) |
| Churn rate | about 26.5% |
| Duplicate customer IDs | 0 |

`customerID` identifies a customer. It is kept for the predictions file and is not used as a model feature.

## Project structure

```text
telco-churn-capstone/
├── data/raw/Telco-Customer-Churn.csv
├── src/
│   ├── data_loader.py      # read the CSV
│   ├── validator.py        # check quality, do not change data
│   ├── preprocessing.py    # clean a copy
│   ├── analysis.py         # churn comparisons
│   ├── model.py            # split and train
│   └── evaluation.py       # accuracy, precision, recall, F1
├── tests/test_churn.py
├── main.py
├── requirements.txt
└── output/predictions.csv  # created when the program runs
```

## Installation

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `scikit-learn`, and `pytest`.

## How to run

```powershell
python main.py --input data/raw/Telco-Customer-Churn.csv
```

The program prints a validation report, exploration findings, and both model scores. It writes `output/predictions.csv` with `customerID`, `actual_churn`, `predicted_churn`, and `churn_probability`.

## How to test

```powershell
pytest
```

The tests use small fake tables. They check a missing column, an invalid churn value, a duplicate customer ID, blank `TotalCharges`, non-negative charges, the prediction columns, and that predictions are only valid classes.

## Validation checks

The validator reports row and column counts, duplicate customer IDs, missing values, invalid `Churn` labels, `TotalCharges` values that are not numeric, negative charges, tenure outside 0–100 months, and unexpected category values.

On this file the status is `REVIEW` because 11 `TotalCharges` values are blank. Those 11 customers have `tenure` 0. There are no duplicate IDs and no invalid churn labels.

## Cleaning decisions

- The raw CSV is never modified. Cleaning is done on a copy.
- `TotalCharges` is stored as text. Blank values are converted to missing numbers, then filled with `0`, because a new customer with tenure 0 has not been billed yet. The column average is not used.
- `Churn` `Yes` becomes `1` and `No` becomes `0`. `1` means the customer churned.
- `customerID` is removed from the feature table and kept beside it for the output file.
- Text columns are encoded later, inside the model step, so training and test data get the same treatment.

## Features and model

Numeric features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`.

Categorical features: gender, partner, dependents, phone and internet services, contract, paperless billing, and payment method.

`customerID` is not a feature.

The split is 70% train and 30% test, with `random_state=42`, and it keeps the same churn proportion in both parts. That is 4,930 training rows and 2,113 test rows.

Two Logistic Regression models were trained on the same split:

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Label encoding | 0.804 | 0.650 | 0.565 | 0.604 |
| One-hot encoding | 0.809 | 0.667 | 0.560 | 0.609 |

One-hot encoding is the model saved into `predictions.csv` because each category gets its own 0/1 column, and its accuracy and F1 were slightly higher. Label encoding forces an order onto categories that do not have a real order.

For the one-hot model, the confusion matrix on the test set is:

|  | Predicted stay | Predicted churn |
| --- | --- | --- |
| Actual stay | 1,395 | 157 |
| Actual churn | 247 | 314 |

A false negative is one of the 247 customers who churned but were predicted to stay. The company would not have offered them help.

Precision means: of the customers flagged as likely to churn, about 66.7% actually churned. Recall means: of the customers who actually churned, the model caught about 56.0%.

## Exploration findings

1. About 26.5% of customers churned.
2. Month-to-month customers churn at 42.7%. Two-year customers churn at 2.8%.
3. Fiber optic customers churn at 41.9%. Customers with no internet service churn at 7.4%.
4. Customers in their first 12 months churn at 47.4%. Customers with 49 or more months churn at 9.5%.
5. Electronic-check customers churn at 45.3%, higher than the other payment methods.
6. Average monthly charges are 74.44 for customers who left and 61.27 for customers who stayed.

## Limitations

- This is one snapshot, not an experiment, so the findings are associations.
- Accuracy alone is misleading. A model that always predicts "stay" would be about 73.5% accurate and would catch no churners.
- Recall is 0.560, so many customers who leave are still missed.
- The optional decision tree, charts, pickle file, and GitHub Actions workflow are not in this version yet.

## Future improvements

- Save the trained model with pickle.
- Add GitHub Actions so pytest runs on every push.
- Compare a Decision Tree with Logistic Regression.
- Add charts for contract, internet service, and tenure.

## Example output

```text
TELCO CUSTOMER CHURN ANALYSIS
Rows loaded: 7043
Overall status: REVIEW
About 26.5% of customers churned.
One-hot encoding accuracy: 0.809
Saved: output/predictions.csv
Prediction rows: 2113
```
