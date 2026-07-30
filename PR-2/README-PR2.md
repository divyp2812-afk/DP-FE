# Data Preprocessing Practical – Missing Value Imputation and Outlier Handling

## 📌 Overview
This practical demonstrates how to preprocess a healthcare dataset by handling missing values, detecting and treating outliers, and preparing a clean dataset for data analysis and machine learning.

---

## 🎯 Objectives

- Identify missing values and summarize them.
- Apply different missing value imputation techniques.
- Detect outliers using multiple methods.
- Handle outliers using suitable techniques.
- Generate a final cleaned dataset.
- Prepare a brief report on data preprocessing results.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- SciPy

---

## 📂 Dataset

**Dataset:** `patient_health_records_1000_rows.csv`

The dataset contains patient health information such as:

- Gender
- Age
- BMI
- Cholesterol
- Region
- Other health-related attributes

---

# Part A – Handling Missing Values

### 1. Missing Value Analysis
- Identified missing values in each column.
- Calculated the percentage of missing values.

### 2. Imputation Techniques

The following methods were applied and compared:

- Mean Imputation
- Most Frequent Imputation
- Categorical Imputation
- Missing Indicator + Random Sample Imputation
- KNN Imputation
- MICE (Multiple Imputation by Chained Equations)

---

# Part B – Handling Outliers

### 3. Outlier Detection Methods

- Z-Score Method
- IQR (Interquartile Range) Method
- Percentile Method

### 4. Outlier Treatment

- Winsorization
- IQR-based Capping

### 5. Comparison

- Dataset shape before and after treatment
- Statistical summary comparison

---

# Part C – Final Clean Dataset

The final dataset contains:

- ✅ Missing values handled
- ✅ Outliers treated
- ✅ Clean numeric data
- ✅ Ready for visualization and machine learning

---

# Brief Report

## 1. Most Effective Imputation Strategy

The **MICE (Multiple Imputation by Chained Equations)** strategy was the most effective because it estimates missing values using relationships between multiple variables. It preserves the overall data pattern better than simple methods such as mean, median, or mode imputation.

## 2. Best Outlier Handling Method

The **IQR Capping (Winsorization)** method preserved data quality best. Instead of removing records, it capped extreme values within acceptable limits, reducing the influence of outliers while retaining all observations.

## 3. Benefits of Data Cleaning

- Missing values were successfully treated.
- Outliers were handled effectively.
- Improved data quality and consistency.
- Increased reliability for analysis and machine learning.
- Reduced bias caused by incomplete or extreme data.

---

# Results

- Successfully identified missing values.
- Applied six imputation techniques.
- Detected outliers using three methods.
- Treated outliers using Winsorization and IQR capping.
- Generated a clean and reliable dataset.

---

# Conclusion

This practical demonstrated important data preprocessing techniques for handling missing values and outliers. Applying advanced imputation methods such as **MICE** and outlier treatment using **IQR Capping/Winsorization** improved the quality, consistency, and reliability of the dataset. The cleaned dataset is now suitable for statistical analysis, visualization, and machine learning applications.
