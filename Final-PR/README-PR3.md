# 📊 PR-3: Holistic Data Preparer

> **A complete data preprocessing and feature engineering project for a Customer Credit Risk dataset.**

This project demonstrates a complete data preparation workflow using **Python, Pandas, NumPy, SciPy, and Scikit-learn**. The notebook covers data analysis, data acquisition, missing-value treatment, outlier detection, categorical encoding, numerical feature engineering, scaling, transformations, and preparation of a final machine-learning-ready dataset.

---

## 📌 Project Overview

The project uses a **Customer Credit Risk Dataset** containing customer demographic, financial, transaction, loan, and credit information.

The main goal is to transform raw data into a clean and structured dataset that can be used for **machine learning and credit-risk analysis**.

### Main workflow

```text
Raw Data
   ↓
Data Acquisition
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Outlier Detection & Treatment
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Numerical Encoding
   ↓
Feature Scaling
   ↓
Feature Transformation
   ↓
New Feature Construction
   ↓
Final Cleaned Dataset
   ↓
Machine Learning Ready
```

---

# 📚 Table of Contents

- [1. Conceptual Foundation](#1-conceptual-foundation)
- [2. Tensors with NumPy](#2-tensors-with-numpy)
- [3. Data Acquisition](#3-data-acquisition)
- [4. Dataset Understanding](#4-dataset-understanding)
- [5. Data Profiling](#5-data-profiling)
- [6. Missing Value Handling](#6-missing-value-handling)
- [7. Outlier Handling](#7-outlier-handling)
- [8. Feature Engineering](#8-feature-engineering)
- [9. Categorical Encoding](#9-categorical-encoding)
- [10. Numerical Feature Encoding](#10-numerical-feature-encoding)
- [11. Feature Scaling](#11-feature-scaling)
- [12. Feature Transformation](#12-feature-transformation)
- [13. Feature Construction](#13-feature-construction)
- [14. Final Dataset](#14-final-dataset)
- [15. Final Report](#15-final-report)
- [16. Technologies Used](#16-technologies-used)
- [17. Project Structure](#17-project-structure)
- [18. Conclusion](#18-conclusion)

---

# 1. 🧠 Conceptual Foundation

## What is Data Analysis?

Data analysis is the process of **collecting, cleaning, organizing, examining, and interpreting data** to discover useful information and support decision-making.

### Steps of Data Analysis

1. Collect data
2. Clean the data
3. Explore the data
4. Analyze the data
5. Visualize results
6. Make decisions based on findings

---

## How to Plan a Data Science Project

The notebook follows a structured data-science workflow:

1. Define the business problem
2. Collect data
3. Clean and preprocess the data
4. Explore data using statistics and visualization
5. Build machine-learning models
6. Evaluate model performance
7. Deploy the model
8. Monitor and improve the model

---

## How to Frame a Machine Learning Problem

A machine-learning problem is created by converting a real-world/business problem into a data-driven problem.

Important steps include:

- Identify the problem
- Define the target variable
- Select input features
- Choose the ML type:
  - Classification
  - Regression
  - Clustering
- Prepare the data
- Select evaluation metrics
- Train and test the model

For this dataset, **`default_flag`** is the binary target variable.

---

# 2. 🔢 Tensors with NumPy

A tensor is a **multi-dimensional array** that generalizes scalars, vectors, and matrices.

NumPy represents tensors using `numpy.ndarray`.

### Scalar

```python
import numpy as np

scalar = np.array(5)
print(scalar)
print(scalar.ndim)
```

### Vector

```python
vector = np.array([1, 2, 3])
print(vector)
print(vector.ndim)
```

### Matrix

```python
matrix = np.array([[1, 2], [3, 4]])
print(matrix)
print(matrix.ndim)
```

### 3D Tensor

```python
tensor_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print(tensor_3d)
print(tensor_3d.ndim)
```

---

# 3. 📥 Data Acquisition

The notebook demonstrates importing data from multiple sources.

## CSV

```python
import pandas as pd

Data = pd.read_csv("customer_credit_risk_dataset.csv")
Data.head()
```

## JSON

```python
import json

with open("customer_metadata.json", "r") as f:
    customer_data = json.load(f)

print(customer_data)
```

## API

The notebook also demonstrates obtaining JSON data from an API using `requests`.

```python
import requests
import pandas as pd

url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)
api_df = pd.DataFrame(response.json())

display(api_df.head())
```

---

# 4. 🔍 Dataset Understanding

The main dataset contains **5,000 records and 15 original columns**.

### Original columns

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `gender` | Customer gender |
| `region` | Customer region |
| `education_level` | Education category |
| `employment_type` | Employment category |
| `annual_income` | Annual income |
| `loan_amount` | Loan amount |
| `loan_purpose` | Purpose of loan |
| `credit_score` | Credit score |
| `repayment_history` | Repayment history measure |
| `transaction_count` | Number of transactions |
| `spending_ratio` | Customer spending ratio |
| `join_date` | Customer joining date |
| `default_flag` | Binary default target |

The notebook uses:

```python
Data.info()
Data.describe()
```

to understand data types, missing values, distributions, and descriptive statistics.

---

# 5. 📋 Data Profiling

A profiling report was generated to examine the quality of the dataset.

```python
from pandas_profiling import ProfileReport

profile = ProfileReport(
    Data,
    title="Customer Credit Risk Dataset Report"
)

profile.to_file("customer_credit_risk_report.html")
```

The profiling stage helps identify:

- Missing values
- Variable types
- Statistical summaries
- Data quality issues
- Potential relationships and unusual values

---

# 6. 🧹 Missing Value Handling

The dataset initially contained missing values in variables such as:

- `age`
- `gender`
- `employment_type`
- `annual_income`
- `loan_amount`
- `credit_score`

The notebook demonstrates several missing-value techniques.

## Simple Imputation

### Mean

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
Data["annual_income"] = imputer.fit_transform(
    Data[["annual_income"]]
)
```

### Median

```python
imputer = SimpleImputer(strategy="median")

Data["loan_amount"] = imputer.fit_transform(
    Data[["loan_amount"]]
)
```

### Most Frequent Category

```python
imputer = SimpleImputer(strategy="most_frequent")

categorical_cols = Data.select_dtypes(
    include=["object", "category"]
).columns

Data[categorical_cols] = imputer.fit_transform(
    Data[categorical_cols]
)
```

### Iterative Imputer / MICE

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)

Data[["annual_income"]] = imputer.fit_transform(
    Data[["annual_income"]]
)
```

### KNN Imputer

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

Data[["annual_income"]] = imputer.fit_transform(
    Data[["annual_income"]]
)
```

### Complete Case Analysis

```python
Data.dropna(inplace=True)
```

### Effectiveness

The final dataset was checked using:

```python
Data.isnull().sum()
```

The final output showed **0 missing values** across the resulting dataset.

---

# 7. 📉 Outlier Handling

The notebook demonstrates four approaches to detecting/treating outliers.

## 7.1 Z-Score

```python
from scipy import stats

z_scores = stats.zscore(Data["annual_income"])

Data = Data[abs(z_scores) < 3]
```

A value with an absolute Z-score greater than approximately 3 can be considered an extreme observation.

---

## 7.2 IQR Method

The Interquartile Range is:

```text
IQR = Q3 - Q1
```

Potential outliers are identified using:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Notebook implementation:

```python
Q1 = Data["loan_amount"].quantile(0.25)
Q3 = Data["loan_amount"].quantile(0.75)

IQR = Q3 - Q1

Data = Data[
    (Data["loan_amount"] >= Q1 - 1.5 * IQR) &
    (Data["loan_amount"] <= Q3 + 1.5 * IQR)
]
```

---

## 7.3 Percentile Method

```python
lower_percentile = Data["loan_amount"].quantile(0.01)
upper_percentile = Data["loan_amount"].quantile(0.99)

Data = Data[
    (Data["loan_amount"] >= lower_percentile) &
    (Data["loan_amount"] <= upper_percentile)
]
```

---

## 7.4 Winsorization

Winsorization limits extreme observations rather than simply deleting them.

```python
from scipy.stats.mstats import winsorize

Data["loan_amount"] = winsorize(
    Data["loan_amount"],
    limits=[0.01, 0.01]
)
```

---

# 8. 🛠️ Feature Engineering

Feature engineering creates new variables from existing variables to provide additional information to machine-learning models.

## Age Groups

```python
Data["age_group"] = pd.cut(
    Data["age"],
    bins=[18, 30, 45, 60, 100],
    labels=["Young", "Adult", "Middle", "Senior"]
)
```

## Region + Age Interaction

```python
Data["region_age"] = (
    Data["region"].astype(str)
    + "_"
    + Data["age_group"].astype(str)
)
```

Example:

```text
South + Middle → South_Middle
South + Young  → South_Young
North + Adult  → North_Adult
```

---

# 9. 🏷️ Categorical Encoding

## Ordinal Encoding

Education level was encoded into numerical categories.

```python
Data["education_encoded"] = (
    Data["education_level"]
    .astype("category")
    .cat.codes
)
```

---

## Label Encoding

Gender was converted into numerical category codes.

```python
Data["gender_encoded"] = (
    Data["gender"]
    .astype("category")
    .cat.codes
)
```

---

## One-Hot Encoding

Categorical columns were converted into binary indicator variables.

```python
cat_cols = Data.select_dtypes(
    include=["object", "category"]
).columns

Data = pd.get_dummies(
    Data,
    columns=cat_cols,
    drop_first=True
)
```

This resulted in a large number of encoded features because interaction variables such as `region_age` were also encoded.

---

# 10. 🔢 Numerical Feature Encoding

The notebook demonstrates several methods.

## Binning

Income was divided into five intervals.

```python
Data["income_binned"] = pd.cut(
    Data["annual_income"],
    bins=5,
    labels=False
)
```

## Binarization

A binary income flag was created using the median.

```python
Data["income_flag"] = (
    Data["annual_income"]
    > Data["annual_income"].median()
).astype(int)
```

## Quantile Binning

```python
Data["income_quantile"] = pd.qcut(
    Data["annual_income"],
    q=5,
    labels=False,
    duplicates="drop"
)
```

## K-Means Binning

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

Data["income_kmeans"] = kmeans.fit_predict(
    Data[["annual_income"]]
)
```

---

# 11. ⚖️ Feature Scaling

Different scaling methods were applied to `annual_income`.

## Standardization / Z-Score Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Data["annual_income_scaled"] = (
    scaler.fit_transform(Data[["annual_income"]])
)
```

Standardization produces a feature with approximately:

```text
Mean = 0
Standard Deviation = 1
```

---

## Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

Data["annual_income_minmax"] = (
    minmax_scaler.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## MaxAbs Scaling

```python
from sklearn.preprocessing import MaxAbsScaler

maxabs_scaler = MaxAbsScaler()

Data["annual_income_maxabs"] = (
    maxabs_scaler.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

Data["annual_income_robust"] = (
    robust_scaler.fit_transform(
        Data[["annual_income"]]
    )
)
```

Robust scaling is useful when a variable contains extreme values because it uses statistics based on the median and interquartile range.

---

# 12. 🔄 Feature Transformation

The notebook applies several mathematical transformations to `annual_income`.

## Log Transformation

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1
)

Data["annual_income_log"] = (
    log_transformer.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Reciprocal Transformation

```python
reciprocal_transformer = FunctionTransformer(
    func=np.reciprocal,
    inverse_func=np.reciprocal
)

Data["annual_income_reciprocal"] = (
    reciprocal_transformer.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Square Root Transformation

```python
sqrt_transformer = FunctionTransformer(
    func=np.sqrt,
    inverse_func=np.square
)

Data["annual_income_sqrt"] = (
    sqrt_transformer.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Box-Cox Transformation

```python
from sklearn.preprocessing import PowerTransformer

box_cox_transformer = PowerTransformer(
    method="box-cox"
)

Data["annual_income_boxcox"] = (
    box_cox_transformer.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Yeo-Johnson Transformation

```python
yeo_johnson_transformer = PowerTransformer(
    method="yeo-johnson"
)

Data["annual_income_yeojohnson"] = (
    yeo_johnson_transformer.fit_transform(
        Data[["annual_income"]]
    )
)
```

---

## Column Transformer

The notebook also demonstrates applying different preprocessing transformations to columns using `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

column_transformer = ColumnTransformer(
    transformers=[
        (
            "boxcox",
            PowerTransformer(method="box-cox"),
            ["annual_income"]
        ),
        (
            "yeojohnson",
            PowerTransformer(method="yeo-johnson"),
            ["annual_income"]
        )
    ]
)

df = column_transformer.fit_transform(Data)
```

---

# 13. 🧩 Feature Construction

## Debt-to-Income Ratio

A new financial feature was constructed:

```python
Data["debt_to_income_ratio"] = (
    Data["loan_amount"]
    / Data["annual_income"]
)
```

### Formula

```text
Debt-to-Income Ratio =
Loan Amount / Annual Income
```

This feature represents the loan amount relative to the customer's annual income.

---

## Average Transaction Feature

The notebook constructs:

```python
Data["avg_transaction"] = (
    Data["spending_ratio"]
    / Data["transaction_count"]
)
```

> **Note:** In this notebook, this is named `avg_transaction`. It uses `spending_ratio / transaction_count`; it should not be interpreted as a true monthly transaction rate unless a time/tenure variable is included.

---

## Spending-to-Income Ratio

```python
Data["spending_to_income_ratio"] = (
    Data["spending_ratio"]
    / Data["annual_income"]
)
```

This feature relates the customer's spending ratio to annual income.

---

# 14. 📅 Date Feature Extraction

The `join_date` variable was converted into a datetime format.

```python
Data["join_date"] = pd.to_datetime(
    Data["join_date"],
    errors="coerce"
)
```

New date features were created:

```python
Data["year"] = Data["join_date"].dt.year
Data["month"] = Data["join_date"].dt.month
Data["day"] = Data["join_date"].dt.day
Data["weekday"] = Data["join_date"].dt.weekday
```

This converts a date into useful numerical features for analysis and modeling.

---

# 15. ✅ Final Dataset

The notebook created a final cleaned and transformed dataset.

### Final shape

```text
Rows:       4,513
Features:   2,292
```

Therefore:

```text
Final Dataset Shape = (4513, 2292)
```

The final dataset was checked for:

- Missing values
- Data types
- Transformed variables
- Engineered features

The notebook reports **0 missing values** in the final dataset.

### Save the final dataset

```python
final_data = Data.copy()

final_data.to_csv(
    "final_cleaned_transformed_dataset.csv",
    index=False
)
```

---

# 16. 📝 Final Report

## Missing Value Strategy

Missing values were identified using `isnull().sum()`.

The notebook demonstrates:

- Mean imputation
- Median imputation
- Most-frequent category imputation
- Iterative Imputation
- KNN Imputation
- Complete-case analysis

The final dataset was checked and contained no missing values.

---

## Outlier Handling

The notebook demonstrates:

- Z-score
- IQR
- Percentile filtering
- Winsorization

These methods were used to identify and reduce the influence of extreme observations.

---

## Encoding

The project demonstrates:

- Ordinal encoding
- Label encoding
- One-hot encoding

These methods convert categorical information into forms that can be processed by machine-learning algorithms.

---

## Scaling

The project demonstrates:

- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler

Scaling helps numerical variables exist on comparable scales.

---

## Transformation

The project demonstrates:

- Log transformation
- Reciprocal transformation
- Square-root transformation
- Box-Cox transformation
- Yeo-Johnson transformation
- ColumnTransformer

These transformations can help modify distributions and make numerical variables more suitable for modeling.

---

## Feature Engineering

Important constructed features include:

- `age_group`
- `region_age`
- `education_encoded`
- `gender_encoded`
- `income_binned`
- `income_flag`
- `income_quantile`
- `income_kmeans`
- `debt_to_income_ratio`
- `avg_transaction`
- `spending_to_income_ratio`
- Date-based features such as `year`, `month`, `day`, and `weekday`

---

# 17. 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| SciPy | Statistical methods and outlier handling |
| Scikit-learn | Imputation, encoding, scaling, transformation, clustering |
| JSON | JSON data handling |
| Requests | API data acquisition |
| Pandas Profiling | Data quality profiling |
| Jupyter Notebook | Interactive development |

---

# 18. 📁 Project Structure

```text
PR-3/
│
├── 📓 PR-3(1).ipynb
├── 📊 customer_credit_risk_dataset.csv
├── 📄 customer_metadata.json
├── 📑 customer_credit_risk_report.html
├── 📊 final_cleaned_transformed_dataset.csv
└── 📘 README.md
```

---

# 19. 🎯 Learning Outcomes

After completing this project, the following concepts were practiced:

- Data analysis
- Data science project planning
- Machine-learning problem framing
- Tensor basics
- CSV, JSON, and API data acquisition
- Dataset exploration
- Data profiling
- Missing-value treatment
- Outlier detection
- Outlier treatment
- Feature engineering
- Categorical encoding
- Numerical encoding
- Feature scaling
- Feature transformation
- Date feature extraction
- Feature construction
- Final dataset preparation

---

# 20. 🏁 Conclusion

This project provides a complete **data preprocessing pipeline** for a Customer Credit Risk dataset.

Starting with raw customer data, the project explores the dataset, identifies data-quality issues, handles missing values, detects and treats outliers, converts categorical variables into numerical representations, applies scaling and mathematical transformations, and constructs additional features.

The final result is a **4,513 × 2,292** cleaned and transformed dataset with **no missing values reported**, making it suitable as a prepared dataset for subsequent machine-learning modeling.

> **PR-3 successfully demonstrates the complete journey from raw data to a machine-learning-ready dataset.** 🚀

---

## 👤 Project Information

**Project:** PR-3 – Holistic Data Preparer  
**Domain:** Customer Credit Risk  
**Language:** Python  
**Environment:** Jupyter Notebook  
**Target Variable:** `default_flag`

---

⭐ **If this project helped you understand data preprocessing, consider adding it to your portfolio/GitHub repository.**
