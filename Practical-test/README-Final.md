# 🛒 Customer Purchase Propensity — Data Cleaning & Feature Engineering

<p align="center">
  <b>A complete practical data preprocessing and feature engineering pipeline</b><br>
  <i>Cleaning • EDA • Imputation • Outliers • Encoding • Scaling • Feature Engineering</i>
</p>

---

## 📌 Project Overview

This project focuses on preparing **customer purchase data** for a machine-learning workflow.

The pipeline combines customer, transaction, and product information, explores the combined dataset, handles data-quality problems, creates useful features, transforms numerical and categorical variables, and exports a processed dataset ready for further analysis or model development.

> **Goal:** Build a clean and feature-engineered dataset that can support a **binary purchase-propensity classification problem**.

---

## 🎯 Objectives

- Import data from multiple sources.
- Combine customer, transaction, and product datasets.
- Perform exploratory data analysis (EDA).
- Identify skewness and potential outliers.
- Apply different missing-value imputation techniques.
- Detect outliers using multiple methods.
- Convert and engineer date/time features.
- Handle mixed variables such as product IDs.
- Apply categorical encoding techniques.
- Scale numerical features using multiple approaches.
- Create transformed and binned features.
- Generate a final processed CSV file.
- Document the complete preprocessing workflow.

---

## 🗂️ Project Structure

```text
Customer-Purchase-Propensity/
│
├── 📓 Final.ipynb
├── 🐍 feature_pipeline.py
├── 📄 customers.csv
├── 🧾 transactions.json
├── 🗃️ products.db
├── 📝 products.sql
├── 📊 processed_customer_data.csv
├── 📈 data_profiling_report.html
└── 📖 README.md
```

---

## 📥 Data Sources

The project works with multiple data sources:

| Source | Format | Purpose |
|---|---|---|
| `customers.csv` | CSV | Customer information |
| `transactions.json` | JSON | Transaction information |
| `products.db` | SQLite | Product information |
| `products.sql` | SQL | Product table definition/sample data |

The transaction data contains fields such as `transaction_id`, `customer_id`, `product_id`, `amount`, `payment_mode`, and `date`.

---

## 🔗 Data Integration

The datasets are merged using their relevant keys:

```text
customers.csv
      │
      │ customer_id
      ▼
transactions.json
      │
      │ product_id
      ▼
products.db
```

### Merge logic

```python
merged_df = pd.merge(
    customer_df,
    transaction_df,
    on="customer_id",
    how="inner"
)

merged_df = pd.merge(
    merged_df,
    product_df,
    on="product_id",
    how="inner"
)
```

---

# 🔍 Exploratory Data Analysis

## 1. Dataset Understanding

The combined dataset is explored using:

```python
merged_df.info()
merged_df.describe()
merged_df.head()
```

This helps understand:

- Dataset structure
- Data types
- Numerical statistics
- Number of observations
- Feature names

---

## 2. Univariate Analysis

### 📊 Histograms

Numerical columns are visualized using histograms to understand their distributions.

```python
merged_df.select_dtypes(include="number").hist(
    figsize=(12, 8),
    bins=10
)
```

### 📐 Skewness

Skewed numerical variables are identified using:

```python
skewness = merged_df.select_dtypes(
    include="number"
).skew()

print(skewness)
```

### 🤖 Automated Profiling

The project also uses **YData Profiling** to generate an automated HTML report.

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(
    merged_df,
    title="Automated Data Analysis Report"
)

profile.to_file("data_profiling_report.html")
```

---

# 📊 Bivariate & Multivariate Analysis

## Income vs Transaction Amount

```python
plt.scatter(
    merged_df["income"],
    merged_df["amount"]
)
```

This helps visually explore the relationship between customer income and transaction amount.

## 🔥 Correlation Heatmap

```python
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
```

## 🔗 Pairplot

The following numerical features are explored together:

```text
age
income
amount
price
stock
```

## 📋 Grouped Statistics

Average income, transaction amount, and price are grouped by product category.

---

# 🧹 Missing Data Handling

Multiple techniques were demonstrated.

### 1. Simple Imputer

**Numerical data:**

```python
SimpleImputer(strategy="median")
```

**Categorical data:**

```python
SimpleImputer(strategy="most_frequent")
```

### 2. Most Frequent Imputation

Mode-based filling is used for categorical variables.

### 3. Missing Indicator + Random Sample

A missing indicator is created:

```python
merged_df["income_missing"] = (
    merged_df["income"].isnull().astype(int)
)
```

Missing income values can then be replaced using random samples from observed values.

### 4. KNN Imputer

```python
KNNImputer(n_neighbors=5)
```

KNN uses information from nearby observations to estimate missing numerical values.

### 5. MICE / Iterative Imputation

```python
IterativeImputer(
    max_iter=10,
    random_state=42
)
```

This demonstrates chained, multivariate imputation for correlated numerical features.

### 6. Complete Case Analysis

Rows containing missing values are removed for comparison:

```python
complete_df = merged_df.dropna()
```

---

# 🚨 Outlier Detection

Three approaches were explored.

## Z-Score

Values with an absolute Z-score greater than approximately `3` are treated as potential outliers.

```python
abs(z_score) > 3
```

## IQR Method

```text
IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

## Percentile Method

The project checks extreme values using the:

```text
1st percentile
99th percentile
```

---

# 📅 Date & Time Features

The transaction `date` column is converted into datetime:

```python
merged_df["date"] = pd.to_datetime(
    merged_df["date"]
)
```

A new feature is created:

```python
days_since_last_purchase
```

This represents the number of days between the transaction date and the current date.

---

# 🔤 Mixed Variables

Product IDs such as:

```text
P001
P002
P003
```

contain both letters and numbers.

A numerical product feature is created:

```python
merged_df["product_number"] = (
    merged_df["product_id"]
    .str.replace("P", "")
    .astype(int)
)
```

Example:

| Product ID | Product Number |
|---|---:|
| P001 | 1 |
| P002 | 2 |
| P003 | 3 |

---

# 🏷️ Categorical Encoding

## Label Encoding

Categorical values such as gender can be converted into numerical labels.

```python
LabelEncoder()
```

## One-Hot Encoding

Categorical variables can be converted into separate binary columns:

```python
pd.get_dummies(
    merged_df,
    dtype=int
)
```

## Ordinal Encoding

Ordinal encoding is demonstrated when a categorical variable has a meaningful order.

> Note: The provided dataset does not contain a natural ordinal field such as education level or satisfaction level, so ordinal encoding should only be applied when an appropriate ordered variable is available.

---

# 📦 Numerical Feature Binning

Income is divided into groups:

```python
merged_df["income_equal_bin"] = pd.cut(
    merged_df["income"],
    bins=3,
    labels=["Low", "Medium", "High"]
)
```

This creates:

```text
Low
Medium
High
```

income groups.

---

# ⚖️ Feature Scaling

Several scaling techniques were explored.

| Scaler | Main Use |
|---|---|
| StandardScaler | Standardization |
| MinMaxScaler | Scale to 0–1 range |
| MaxAbsScaler | Scale by maximum absolute value |
| RobustScaler | Useful with outliers |
| Normalizer | Normalize observations |

### StandardScaler

```python
StandardScaler()
```

### MinMaxScaler

```python
MinMaxScaler()
```

### MaxAbsScaler

```python
MaxAbsScaler()
```

### RobustScaler

```python
RobustScaler()
```

### Normalizer

```python
Normalizer()
```

---

# 🧩 ColumnTransformer

`ColumnTransformer` allows different preprocessing methods to be applied to different columns simultaneously.

Example:

```python
ColumnTransformer(
    transformers=[
        ("standard", StandardScaler(), ["age", "income"]),
        ("minmax", MinMaxScaler(), ["amount", "price", "stock"])
    ]
)
```

This is useful when different features require different preprocessing strategies.

---

# 🔄 Feature Transformations

## Log Transformation

```python
np.log1p(x)
```

Useful for reducing right-skewed distributions.

## Reciprocal Transformation

```python
1 / (x + 1)
```

## Square Root Transformation

```python
np.sqrt(x)
```

These transformations were demonstrated using the income feature.

---

# 📈 Power Transformations

The project also demonstrates:

- **Box-Cox Transformation**
- **Yeo-Johnson Transformation**

These transformations can help make numerical variables more suitable for statistical or machine-learning algorithms.

---

# 🛍️ Purchase Frequency & Binarization

A `frequent_buyer` feature is created from customer purchase frequency.

```python
purchase_count = merged_df.groupby(
    "customer_id"
)["date"].transform("count")

merged_df["frequent_buyer"] = (
    purchase_count > 2
).astype(int)
```

### Rule

```text
More than 2 purchases → 1
2 or fewer purchases → 0
```

---

# 💾 Final Output

The processed dataset is exported as:

```text
processed_customer_data.csv
```

using:

```python
merged_df.to_csv(
    "processed_customer_data.csv",
    index=False
)
```

An automated profiling report is also generated as:

```text
data_profiling_report.html
```

---

# 📝 Final Summary

## Techniques Used

The project uses:

- Data integration
- Exploratory Data Analysis
- Histograms
- Skewness analysis
- Correlation analysis
- Pairplots
- Heatmaps
- Grouped statistics
- Simple Imputation
- Most Frequent Imputation
- Missing Indicators
- Random Sample Imputation
- KNN Imputation
- MICE / Iterative Imputation
- Complete Case Analysis
- Z-Score outlier detection
- IQR outlier detection
- Percentile outlier detection
- Date conversion
- Date-based feature engineering
- Mixed-variable handling
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding
- Binning
- Binarization
- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- Normalizer
- ColumnTransformer
- FunctionTransformer
- Box-Cox
- Yeo-Johnson
- Feature engineering

## Biggest Raw-Data Issues

The main data-quality challenges were:

- Missing values
- Potential outliers
- Categorical variables
- Numerical variables with different scales
- Date fields requiring conversion
- Mixed-format identifiers
- Some requested assignment fields were not present in the provided dataset

## Best Imputation & Scaling

For categorical missing values, **Most Frequent Imputation** is a practical choice.

For numerical data with potential outliers, **RobustScaler** is useful because it is less sensitive to extreme values.

For multivariate missing numerical data, **KNN Imputer** and **Iterative/MICE imputation** provide more advanced alternatives.

---

# 🚀 How to Run

### 1. Install required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ydata-profiling requests
```

### 2. Place the input files in the project folder

```text
customers.csv
transactions.json
products.db
```

### 3. Open the notebook

```text
Final.ipynb
```

### 4. Run the cells

Run the preprocessing and feature-engineering cells in order.

### 5. Check the generated files

```text
processed_customer_data.csv
data_profiling_report.html
```

---

# 📚 Technologies Used

```text
🐍 Python
📊 Pandas
🔢 NumPy
📈 Matplotlib
🎨 Seaborn
🤖 Scikit-learn
📐 SciPy
📋 YData Profiling
🗃️ SQLite
🌐 Requests
```

---

# 👨‍💻 Project Type

**Practical Data Analysis & Machine Learning Preprocessing Project**

**Focus:** Customer Purchase Propensity • Data Cleaning • EDA • Feature Engineering • Preprocessing

---

<p align="center">
  <b>✨ Clean Data → Better Features → Better Models ✨</b>
</p>
