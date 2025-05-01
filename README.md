Carbon Footprint Prediction Model

This project trains a stacked ensemble model to predict household carbon footprints using structured data and engineered features that reflect lifestyle, energy behavior, and housing 
characteristics.

Overview:
The code:
- Loads and cleans a tabular dataset (CSV format)
- Handles missing values and outliers
- Performs targeted feature engineering (e.g., electricity use × solar adoption)
- Builds a stacked model using XGBoost, Ridge Regression, and SVR
- Evaluates model performance with K-Fold cross-validation
- Generates final predictions for submission

Feature Engineering

Includes domain-informed interactions like:
- `electricity_x_no_solar`: electricity usage weighted by lack of solar
