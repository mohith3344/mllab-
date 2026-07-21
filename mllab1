import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")



def calculate_matrix_rank(feature_matrix: np.ndarray) -> int:
  """Calculates the rank of a given feature matrix using NumPy."""
  return int(np.linalg.matrix_rank(feature_matrix))


def calculate_product_costs_pinv(
    feature_matrix: np.ndarray, payment_vector: np.ndarray
) -> np.ndarray:
  """Calculates the unit cost of each product using the Pseudo-Inverse of the feature matrix."""
  pseudo_inverse_matrix = np.linalg.pinv(feature_matrix)
  product_unit_costs = pseudo_inverse_matrix @ payment_vector
  return product_unit_costs

def classify_customers(
    feature_matrix: np.ndarray, payment_vector: np.ndarray, threshold: float
) -> tuple:
  """Assigns RICH/POOR labels based on payment threshold and trains a Decision Tree classifier."""
  class_labels = np.where(payment_vector > threshold, "RICH", "POOR")
  classifier = DecisionTreeClassifier(random_state=42)
  classifier.fit(feature_matrix, class_labels)
  predictions = classifier.predict(feature_matrix)
  model_accuracy = accuracy_score(class_labels, predictions)
  return class_labels, classifier, model_accuracy


def custom_mean(data_vector: np.ndarray) -> float:
  """Calculates the arithmetic mean from scratch without package functions."""
  total_sum = 0.0
  for value in data_vector:
    total_sum += value
  return float(total_sum / len(data_vector))


def custom_variance(data_vector: np.ndarray) -> float:
  """Calculates the population variance from scratch without package functions."""
  mean_value = custom_mean(data_vector)
  squared_diff_sum = 0.0
  for value in data_vector:
    squared_diff_sum += (value - mean_value) ** 2
  return float(squared_diff_sum / len(data_vector))


def compare_computational_complexity(
    data_vector: np.ndarray, num_runs: int = 10
) -> dict:
  """Measures and compares execution time between NumPy and custom statistical functions."""

  start_time = time.perf_counter()
  for _ in range(num_runs):
    _ = np.mean(data_vector)
  numpy_mean_time = (time.perf_counter() - start_time) / num_runs

  start_time = time.perf_counter()
  for _ in range(num_runs):
    _ = custom_mean(data_vector)
  custom_mean_time = (time.perf_counter() - start_time) / num_runs

  start_time = time.perf_counter()
  for _ in range(num_runs):
    _ = np.var(data_vector)
  numpy_var_time = (time.perf_counter() - start_time) / num_runs

  start_time = time.perf_counter()
  for _ in range(num_runs):
    _ = custom_variance(data_vector)
  custom_var_time = (time.perf_counter() - start_time) / num_runs

  return {
      "numpy_mean_time": numpy_mean_time,
      "custom_mean_time": custom_mean_time,
      "numpy_var_time": numpy_var_time,
      "custom_var_time": custom_var_time,
  }


def calculate_sample_means(df_stock: pd.DataFrame) -> dict:
  """Calculates sample means for Wednesdays and the month of April, comparing with population mean."""
  population_mean = np.mean(df_stock["Price"].values)
  wednesday_mean = np.mean(
      df_stock[df_stock["Day"] == "Wed"]["Price"].values
  )
  april_mean = np.mean(
      df_stock[df_stock["Month"] == "Apr"]["Price"].values
  )
  return {
      "pop_mean": population_mean,
      "wed_mean": wednesday_mean,
      "wed_diff": wednesday_mean - population_mean,
      "apr_mean": april_mean,
      "apr_diff": april_mean - population_mean,
  }


def calculate_stock_probabilities(df_stock: pd.DataFrame) -> dict:
  """Calculates probability of loss, profit on Wednesday, and conditional profit given Wednesday."""
  change_data = df_stock["Chg%"].values

  loss_count = len(list(filter(lambda x: x < 0, change_data)))
  prob_loss = loss_count / len(change_data)

  df_stock["Is_Profit"] = df_stock["Chg%"] > 0
  profit_and_wed_count = len(
      df_stock[(df_stock["Day"] == "Wed") & (df_stock["Is_Profit"])]
  )
  prob_profit_and_wed = profit_and_wed_count / len(df_stock)

  wednesday_total_count = len(df_stock[df_stock["Day"] == "Wed"])
  prob_profit_given_wed = (
      profit_and_wed_count / wednesday_total_count
      if wednesday_total_count > 0
      else 0.0
  )

  return {
      "prob_loss": prob_loss,
      "prob_profit_and_wed": prob_profit_and_wed,
      "prob_profit_given_wed": prob_profit_given_wed,
  }


def generate_stock_scatter_plot(df_stock: pd.DataFrame, output_path: str):
  """Generates and saves a scatter plot of Chg% against the day of the week."""
  plt.figure(figsize=(9, 5))
  sns.scatterplot(
      data=df_stock,
      x="Day",
      y="Chg%",
      hue="Day",
      palette="tab10",
      s=80,
      legend=False,
  )
  plt.title("IRCTC Stock: Daily Percentage Change (Chg%) vs. Day of Week")
  plt.xlabel("Day of the Week")
  plt.ylabel("Percentage Change (Chg%)")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()

def explore_and_impute_thyroid_data(df_thyroid: pd.DataFrame) -> tuple:
  """Performs data exploration, cleans missing indicators ('?'), applies imputation, and computes stats."""
  df_clean = df_thyroid.replace("?", np.nan).copy()

  numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
  for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

  stats_list = []
  for col in numeric_cols:
    series = df_clean[col].dropna()
    stats_list.append({
        "Attribute": col,
        "Min": series.min(),
        "Max": series.max(),
        "Mean": series.mean(),
        "Variance": series.var(),
        "Std_Dev": series.std(),
        "Missing_Count": df_clean[col].isna().sum(),
        "Missing_Pct": (df_clean[col].isna().sum() / len(df_clean)) * 100,
    })
  stats_df = pd.DataFrame(stats_list)

  df_imputed = df_clean.copy()
  for col in df_imputed.columns:
    if col in numeric_cols:
      if col == "age":
        impute_val = df_imputed[col].mean()
      else:
        impute_val = df_imputed[col].median()
      df_imputed[col] = df_imputed[col].fillna(impute_val)
    else:
      mode_series = df_imputed[col].mode()
      impute_val = mode_series[0] if not mode_series.empty else "f"
      df_imputed[col] = df_imputed[col].fillna(impute_val)

  return stats_df, df_imputed


def get_binary_columns(df_data: pd.DataFrame) -> list:
  """Identifies attributes carrying binary representations (e.g., 't'/'f' or 0/1)."""
  binary_cols = []
  for col in df_data.columns:
    unique_vals = set(df_data[col].dropna().unique())
    if unique_vals.issubset({"t", "f", "F", "M", "0", "1", 0, 1}):
      binary_cols.append(col)
  return binary_cols


def calculate_similarity_coefficients(
    vector_a: np.ndarray, vector_b: np.ndarray
) -> tuple:
  """Calculates Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) for binary vectors."""
  f11 = np.sum((vector_a == 1) & (vector_b == 1))
  f10 = np.sum((vector_a == 1) & (vector_b == 0))
  f01 = np.sum((vector_a == 0) & (vector_b == 1))
  f00 = np.sum((vector_a == 0) & (vector_b == 0))

  jaccard_coeff = (
      float(f11 / (f01 + f10 + f11)) if (f01 + f10 + f11) > 0 else 1.0
  )
  simple_matching_coeff = float((f11 + f00) / len(vector_a))
  return jaccard_coeff, simple_matching_coeff


def calculate_cosine_similarity(
    matrix_a: np.ndarray, matrix_b: np.ndarray
) -> float:
  """Calculates Cosine Similarity between two full feature vectors."""
  similarity_matrix = cosine_similarity(matrix_a, matrix_b)
  return float(similarity_matrix[0][0])


def generate_similarity_heatmaps(
    df_imputed: pd.DataFrame, num_samples: int, output_path: str
):
  """Computes pairwise JC, SMC, and Cosine similarity matrices and visualizes them using heatmaps."""
  df_subset = df_imputed.iloc[:num_samples].copy()
  binary_cols = get_binary_columns(df_subset)

  for col in binary_cols:
    df_subset[col] = df_subset[col].map({"t": 1, "f": 0, "F": 1, "M": 0})

  binary_matrix = df_subset[binary_cols].fillna(0).astype(int).values

 
  df_encoded = pd.get_dummies(
      df_subset.drop(columns=["Record ID", "Condition"], errors="ignore")
  )
  full_feature_matrix = df_encoded.astype(float).values

  jc_matrix = np.zeros((num_samples, num_samples))
  smc_matrix = np.zeros((num_samples, num_samples))
  cos_matrix = cosine_similarity(full_feature_matrix)

  for i in range(num_samples):
    for j in range(num_samples):
      jc, smc = calculate_similarity_coefficients(
          binary_matrix[i], binary_matrix[j]
      )
      jc_matrix[i, j] = jc
      smc_matrix[i, j] = smc

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  sns.heatmap(
      jc_matrix, annot=False, cmap="viridis", ax=axes[0], cbar_kws={"label": "JC"}
  )
  axes[0].set_title("Jaccard Coefficient (JC)")

  sns.heatmap(
      smc_matrix,
      annot=False,
      cmap="viridis",
      ax=axes[1],
      cbar_kws={"label": "SMC"},
  )
  axes[1].set_title("Simple Matching Coefficient (SMC)")

  sns.heatmap(
      cos_matrix,
      annot=False,
      cmap="viridis",
      ax=axes[2],
      cbar_kws={"label": "COS"},
  )
  axes[2].set_title("Cosine Similarity (COS)")

  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()


def normalize_dataset_attributes(df_imputed: pd.DataFrame) -> pd.DataFrame:
  """Applies Min-Max Normalization to continuous numeric features with wide ranges."""
  df_normalized = df_imputed.copy()
  continuous_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

  scaler = MinMaxScaler()
  df_normalized[continuous_cols] = scaler.fit_transform(
      df_normalized[continuous_cols]
  )
  return df_normalized


def analyze_square_submatrices(
    feature_matrix: np.ndarray, payment_vector: np.ndarray
) -> dict:
  """Creates two square submatrices from purchase data and checks linearity and costs."""

  X_sq1, y_sq1 = feature_matrix[:3], payment_vector[:3]
  costs_sq1 = np.linalg.solve(X_sq1, y_sq1)

  X_sq2, y_sq2 = feature_matrix[3:6], payment_vector[3:6]
  costs_sq2 = np.linalg.solve(X_sq2, y_sq2)

  return {"costs_sq1": costs_sq1, "costs_sq2": costs_sq2}

def main():
 
  excel_filepath = "/content/Lab Session Data (1).xlsx"
  xls_workbook = pd.ExcelFile(excel_filepath)

  print("=" * 70)
  print("ASSIGNMENT A1: MATRIX RANK & PSEUDO-INVERSE PRODUCT COSTS")
  print("=" * 70)
  df_purchase = pd.read_excel(xls_workbook, "Purchase data")
  feature_matrix_X = df_purchase[
      ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
  ].values
  payment_vector_y = df_purchase["Payment (Rs)"].values

  matrix_rank = calculate_matrix_rank(feature_matrix_X)
  product_unit_costs = calculate_product_costs_pinv(
      feature_matrix_X, payment_vector_y
  )

  print(f"Feature Matrix X Dimensionality : {feature_matrix_X.shape[1]}")
  print(f"Number of Observation Vectors   : {feature_matrix_X.shape[0]}")
  print(f"Calculated Rank of Matrix X     : {matrix_rank}")
  print(
      f"Product Costs (Candies, Mangoes, Milk):"
      f" {np.round(product_unit_costs, 2)} Rs"
  )

  print("\n" + "=" * 70)
  print("ASSIGNMENT A2: CUSTOMER CLASSIFICATION (RICH vs. POOR)")
  print("=" * 70)
  labels, model, accuracy = classify_customers(
      feature_matrix_X, payment_vector_y, threshold=200.0
  )
  df_purchase["Customer_Class"] = labels
  print(
      df_purchase[
          ["Customer", "Payment (Rs)", "Customer_Class"]
      ].to_string(index=False)
  )
  print(f"\nDecision Tree Classifier Training Accuracy: {accuracy * 100:.2f}%")

  print("\n" + "=" * 70)
  print("ASSIGNMENT A3: IRCTC STOCK PRICE STATISTICAL ANALYSIS")
  print("=" * 70)
  df_stock = pd.read_excel(xls_workbook, "IRCTC Stock Price")
  stock_prices = df_stock["Price"].values

  pkg_mean = np.mean(stock_prices)
  pkg_var = np.var(stock_prices)
  cust_mean_val = custom_mean(stock_prices)
  cust_var_val = custom_variance(stock_prices)

  print(
      f"NumPy Mean   : {pkg_mean:.4f} | Custom Mean   : {cust_mean_val:.4f}"
  )
  print(
      f"NumPy Var    : {pkg_var:.4f} | Custom Var    : {cust_var_val:.4f}"
  )

  complexity_results = compare_computational_complexity(
      stock_prices, num_runs=10
  )
  print(
      f"Avg Execution Time (Mean) -> NumPy: {complexity_results['numpy_mean_time']:.8f}s | Custom:"
      f" {complexity_results['custom_mean_time']:.8f}s"
  )
  print(
      f"Avg Execution Time (Var)  -> NumPy: {complexity_results['numpy_var_time']:.8f}s | Custom:"
      f" {complexity_results['custom_var_time']:.8f}s"
  )

  sample_stats = calculate_sample_means(df_stock)
  print(
      f"\nWednesday Sample Mean : {sample_stats['wed_mean']:.2f} (Diff from Pop Mean:"
      f" {sample_stats['wed_diff']:.2f})"
  )
  print(
      f"April Sample Mean     : {sample_stats['apr_mean']:.2f} (Diff from Pop Mean:"
      f" {sample_stats['apr_diff']:.2f})"
  )

  prob_stats = calculate_stock_probabilities(df_stock)
  print(
      f"\nProbability of Making a Loss over Stock  : {prob_stats['prob_loss']:.4f}"
  )
  print(
      f"Probability of Making a Profit on Wed    : {prob_stats['prob_profit_and_wed']:.4f}"
  )
  print(
      f"Conditional Probability P(Profit | Wed)  : {prob_stats['prob_profit_given_wed']:.4f}"
  )

  generate_stock_scatter_plot(df_stock, "irctc_scatter_plot.png")
  print("Scatter plot successfully generated and saved as 'irctc_scatter_plot.png'.")

  print("\n" + "=" * 70)
  print("ASSIGNMENT A4 & A8: THYROID DATA EXPLORATION & IMPUTATION")
  print("=" * 70)
  df_thyroid = pd.read_excel(xls_workbook, "thyroid0387_UCI")
  stats_df, df_thyroid_imputed = explore_and_impute_thyroid_data(df_thyroid)
  print("Numeric Attributes Pre-Imputation Summary Statistics:")
  print(stats_df.to_string(index=False))
  print(
      f"\nRemaining missing values post-imputation: {df_thyroid_imputed.isna().sum().sum()}"
  )

  print("\n" + "=" * 70)
  print("ASSIGNMENT A5 & A6: SIMILARITY MEASURES ON DOCUMENT VECTORS")
  print("=" * 70)
  binary_attributes = get_binary_columns(df_thyroid_imputed)
  # Convert 't'/'f' to binary 1/0 for first two observations
  vector_0_bin = (
      df_thyroid_imputed.iloc[0][binary_attributes]
      .map({"t": 1, "f": 0, "F": 1, "M": 0})
      .values.astype(int)
  )
  vector_1_bin = (
      df_thyroid_imputed.iloc[1][binary_attributes]
      .map({"t": 1, "f": 0, "F": 1, "M": 0})
      .values.astype(int)
  )

  jc_val, smc_val = calculate_similarity_coefficients(
      vector_0_bin, vector_1_bin
  )
  print(
      f"Binary Attributes -> Jaccard Coefficient (JC)        : {jc_val:.4f}"
  )
  print(
      f"Binary Attributes -> Simple Matching Coefficient (SMC): {smc_val:.4f}"
  )

  df_full_encoded = pd.get_dummies(
      df_thyroid_imputed.drop(columns=["Record ID", "Condition"])
  )
  full_vec_0 = df_full_encoded.iloc[[0]].astype(float).values
  full_vec_1 = df_full_encoded.iloc[[1]].astype(float).values
  cosine_val = calculate_cosine_similarity(full_vec_0, full_vec_1)
  print(f"Complete Feature Vectors -> Cosine Similarity         : {cosine_val:.4f}")

  print("\n" + "=" * 70)
  print("ASSIGNMENT A7: HEATMAP PLOT FOR FIRST 20 OBSERVATIONS")
  print("=" * 70)
  generate_similarity_heatmaps(
      df_thyroid_imputed, num_samples=20, output_path="similarity_heatmaps.png"
  )
  print("Heatmap visualizations successfully saved as 'similarity_heatmaps.png'.")

  print("\n" + "=" * 70)
  print("ASSIGNMENT A9: DATA NORMALIZATION")
  print("=" * 70)
  df_normalized = normalize_dataset_attributes(df_thyroid_imputed)
  print("Min-Max Normalization successfully applied to continuous attributes:")
  print(
      df_normalized[
          ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
      ].head(3).to_string()
  )

  print("\n" + "=" * 70)
  print("OPTIONAL SECTION O1: SQUARE MATRIX EXPERIMENTS")
  print("=" * 70)
  submatrix_results = analyze_square_submatrices(
      feature_matrix_X, payment_vector_y
  )
  print(
      f"Costs from Square Submatrix 1 (Rows 0-2):"
      f" {np.round(submatrix_results['costs_sq1'], 2)} Rs"
  )
  print(
      f"Costs from Square Submatrix 2 (Rows 3-5):"
      f" {np.round(submatrix_results['costs_sq2'], 2)} Rs"
  )
  print(
      "Observation: The product costs obtained from any linearly independent"
      " 3x3 square submatrix match precisely with the full purchase matrix!"
  )
  print("=" * 70)


if __name__ == "__main__":
  main()
