[HackerEarth Machine Learning Challenge: Earth Day](https://www.hackerearth.com/challenges/new/competitive/hackerearth-machine-learning-challenge-earth-day/)

# Carbon Footprint Prediction Model

This project was created for an online machine learning competition focused on predicting individual household carbon footprints using data and engineered features that reflect lifestyle, energy behavior, and housing-related characteristics. The solution, which placed 12th out of 1000+ participants with an R<sup>2</sup> score of .8830826, uses a stacked ensemble with XGBoost, Ridge Regression, and SVR as base models, and an XGBoost meta-model. The approach involved exploratory data analysis, data preprocessing, feature engineering, base model tuning using Optuna, and final prediction using the meta-model trained on out-of-fold predictions.

## EDA:
For exploratory analysis, histograms and boxplots were used to visualize the spread of continuous variables, assess skewness, identify outliers, etc. This, combined with a table of descriptive 
statistics for each feature, provided initial insights and assisted in eventual feature engineering.

A Pearson correlation heatmap was used to identify relationships between all input features and the target. The results showed that `meat_consumption_kg_per_week`, `natural_gas_therms_per_month`, and 
`vehicle_miles_per_month` had the strongest linear relationships with carbon_footprint.

Scatterplots, including ones that were conditioned on binary variables like solar panel usage, were used to examine potential interactions. For example, plotting `electricity_kwh_per_month` against 
`carbon_footprint` both with and without solar panel usage (`uses_solar_panels`) showed that the combination of `electricity_kwh_per_month` and `uses_solar_panels` was a stronger predictor than each feature independently.


## Data Preprocessing and Feature Engineering:
- Missing values in binary columns were filled with 0.
- For numeric columns, the median was used to fill missing values instead of the mean to reduce sensitivity to outliers.
- The `heating_type` column contained invalid entries, which were replaced with the string `"garbage"` so the rows could be flagged.
- One-hot encoding was applied to categorical variables `heating_type` and `diet_type`.
- All numeric features were standardized using `StandardScaler`.
- During EDA, histograms of several features had revealed suspicious negative values in fields that should have been non-negative (e.g. negative values in `water_usage_liters_per_day`, among other 
features). These were not removed (as doing so degraded model performance) but instead flagged with binary indicators, which improved results. 
- The engineered feature `electricity_x_no_solar`, which combined high electricity usage with a lack of solar panels, was inspired by patterns observed in the EDA. I chose to combine `electricity_kwh_per_month` with `(1 - uses_solar_panels)` rather than `uses_solar_panels` because there was a higher number of samples in which households lacked solar panels.
- Other patterns emerged during the EDA, such as relationships between `house_area_sqft`, `home_insulation_quality`, and `uses_solar` (as well as several other relationships), but implementing some of these features worsened performance – likely due to overfitting, since relatively few samples met all the conditions – so I did not include them in the final model. Some such features likely involved interactions that were too complex for a model trained on a dataset of this size to generalize well. Others had ample datapoints but had relationships that were not statistically significant enough to improve performance.

## Models:
Three base models were used: XGBoost, Ridge Regression, and SVR. This combination was chosen to balance different model tendencies – XGBoost for its successful track record with tabular data and its 
ability to handle complex relationships, Ridge for simple, regularized linear trends, and SVR for cases where allowing small margins of error might lead to better generalization.
Each was tuned using Optuna, with 50–150 trials per model.

Seven-fold cross-validation was used to generate out-of-fold predictions from each base model. These predictions were used as features to train and test both a Ridge Regression meta-model and 
XGBoost meta-model for comparison. Ridge gave more consistent results overall, but after careful tuning, XGBoost performed slightly better and was selected as the final meta-model.
