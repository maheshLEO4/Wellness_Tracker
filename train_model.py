import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy import stats

# 1. Load dataset
df = pd.read_csv("data/wellness_dataset_cleaned.csv")

# 2. Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

# 3. Outlier removal on numeric columns only
z_scores = np.abs(stats.zscore(df[numeric_cols]))
threshold = 3
df_clean_numeric = df[(z_scores < threshold).all(axis=1)]

# Keep non-numeric columns if needed (for now we drop them if any)
df_clean = df_clean_numeric

X = df_clean.drop('wellness_index', axis=1)
y = df_clean['wellness_index']

print(f"✅ Data cleaned: {len(df) - len(df_clean)} outliers removed.")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Pipeline: Scaling + Polynomial Features + Ridge
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])

# 6. Model training
pipeline.fit(X_train, y_train)

# 7. Predictions
y_pred = pipeline.predict(X_test)

# 8. Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("📈 Polynomial Regression (degree=2) with Ridge Regularization")
print(f"Test R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
print(f"RMSE: {rmse:.4f}")

# 9. Save pipeline for deployment
joblib.dump(pipeline, "model.pkl")
print("✅ Model pipeline saved (ready for deployment).")
