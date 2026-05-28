import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

print("🚗 Init: Multi-Feature Car Value Architecture Predictor...")
np.random.seed(7)
sample_volume = 250

engine_displacement = np.random.uniform(1.0, 4.0, sample_volume)
odometer_reading = np.random.randint(5000, 140000, sample_volume)
brand_tier = np.random.choice([1, 2, 3], size=sample_volume, p=[0.2, 0.5, 0.3]) # 1=Luxury, 2=Mid, 3=Economy

# Complex pricing structure using multi-variable calculations
target_valuation = (brand_tier * -180000) + (engine_displacement * 220000) - (odometer_reading * 2.8) + 1200000
target_valuation += np.random.normal(0, 45000, sample_volume)
target_valuation = np.clip(target_valuation, 120000, None)

dataset_compiled = pd.DataFrame({
    'Displacement_Liters': engine_displacement,
    'Odometer_KM': odometer_reading,
    'Brand_Tier_Index': brand_tier,
    'Market_Value_INR': target_valuation
})

X_matrix = dataset_compiled[['Displacement_Liters', 'Odometer_KM', 'Brand_Tier_Index']]
y_vector = dataset_compiled['Market_Value_INR']

X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_vector, test_size=0.2, random_state=7)

# Unique approach: Using StandardScaler normalization pipeline and ExtraTreesRegressor
normalization_scaler = StandardScaler()
X_train_scaled = normalization_scaler.fit_transform(X_train)
X_test_scaled = normalization_scaler.transform(X_test)

inference_engine = ExtraTreesRegressor(n_estimators=150, random_state=7)
inference_engine.fit(X_train_scaled, y_train)
eval_predictions = inference_engine.predict(X_test_scaled)

print(f"🎯 ExtraTrees Regression Index Fit Score: {r2_score(y_test, eval_predictions) * 100:.2f}%")

plt.figure(figsize=(8, 5))
residuals = y_test - eval_predictions
sns.histplot(residuals, kde=True, color='#009688', bins=20)
plt.title('Structural Error Distribution Diagnostics (Residuals Map)', fontsize=11, fontweight='bold')
plt.xlabel('Prediction Margin Deviation Value (INR)')
plt.tight_layout()
plt.savefig('valuation_engine_residual_diagnostics.png')
print("💾 Artifact generated successfully: 'valuation_engine_residual_diagnostics.png'")
plt.show()
