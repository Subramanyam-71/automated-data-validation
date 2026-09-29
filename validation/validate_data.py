import pandas as pd
import pandera.pandas as pa
from scipy.stats import ks_2samp


# ==========================================
# 1. LOAD DATA
# ==========================================

train_df = pd.read_csv("../data/train_data.csv")
incoming_df = pd.read_csv("../data/incoming_data.csv")

print("========================================")
print("     AUTOMATED DATA VALIDATION")
print("========================================")


# ==========================================
# 2. SCHEMA VALIDATION
# ==========================================

print("\n[1] Checking Schema...")

schema = pa.DataFrameSchema({
    "customer_id": pa.Column(int),
    "age": pa.Column(int),
    "income": pa.Column(int),
    "spending_score": pa.Column(int)
})

schema_passed = True

try:
    schema.validate(incoming_df)
    print("✅ Schema validation PASSED")

except pa.errors.SchemaError as e:
    print("❌ Schema validation FAILED")
    print(e)
    schema_passed = False


# ==========================================
# 3. MISSING VALUE CHECK
# ==========================================

print("\n[2] Checking Missing Values...")

missing_values = incoming_df.isnull().sum().sum()

if missing_values == 0:
    print("✅ Missing value check PASSED")
    missing_passed = True

else:
    print(f"❌ Missing value check FAILED")
    print(f"Found {missing_values} missing value(s)")
    missing_passed = False


# ==========================================
# 4. DUPLICATE CHECK
# ==========================================

print("\n[3] Checking Duplicate Records...")

duplicates = incoming_df.duplicated().sum()

if duplicates == 0:
    print("✅ Duplicate check PASSED")
    duplicate_passed = True

else:
    print("❌ Duplicate check FAILED")
    print(f"Found {duplicates} duplicate record(s)")
    duplicate_passed = False


# ==========================================
# 5. OUTLIER CHECK
# ==========================================

print("\n[4] Checking Outliers...")


def check_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return len(outliers)


age_outliers = check_outliers(incoming_df, "age")
income_outliers = check_outliers(incoming_df, "income")
score_outliers = check_outliers(incoming_df, "spending_score")

total_outliers = age_outliers + income_outliers + score_outliers

if total_outliers == 0:
    print("✅ Outlier check PASSED")
    outlier_passed = True

else:
    print("❌ Outlier check FAILED")
    print(f"Found {total_outliers} potential outlier value(s)")
    outlier_passed = False


# ==========================================
# 6. DISTRIBUTION CHANGE CHECK
# ==========================================

print("\n[5] Checking Distribution Changes...")

distribution_passed = True

columns_to_check = [
    "age",
    "income",
    "spending_score"
]

for column in columns_to_check:

    statistic, p_value = ks_2samp(
        train_df[column],
        incoming_df[column]
    )

    print(f"{column}: p-value = {p_value:.4f}")

    # p-value < 0.05 means significant distribution change
    if p_value < 0.05:
        print(f"❌ Distribution change detected in {column}")
        distribution_passed = False


if distribution_passed:
    print("✅ Distribution check PASSED")


# ==========================================
# 7. FINAL DATA QUALITY GATE
# ==========================================

print("\n========================================")
print("             FINAL RESULT")
print("========================================")

all_checks_passed = (
    schema_passed
    and missing_passed
    and duplicate_passed
    and outlier_passed
    and distribution_passed
)

if all_checks_passed:

    print("✅ DATA VALIDATION PASSED")
    print("Dataset is allowed to enter the ML pipeline.")

else:

    print("❌ DATA VALIDATION FAILED")
    print("Dataset is BLOCKED from entering the ML pipeline.")

print("========================================")