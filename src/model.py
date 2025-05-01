
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from scipy.stats import zscore

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_data_folder = os.path.join(project_root, 'data')

# Load data
train_df = pd.read_csv(os.path.join(path_to_data_folder, 'train.csv'))
test = pd.read_csv(os.path.join(path_to_data_folder, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(path_to_data_folder, 'sample_submission.csv'))

#DATA CLEANING

X = train_df.drop(['ID', 'carbon_footprint'], axis=1) #drop the output column
y = train_df['carbon_footprint']

test_ids = test['ID'] #save IDs for final submission
test = test.drop('ID', axis=1) #drop 'ID' before modeling




#Fix null values in binary columns

binary_cols = ['recycles_regularly', 'composts_organic_waste',
               'energy_efficient_appliances', 'smart_thermostat_installed', 'owns_pet']
X[binary_cols] = X[binary_cols].fillna(0) #fill missing binary values with 0
test[binary_cols] = test[binary_cols].fillna(0)


#Fix null values in numerical columns

# For house_area_sqft

X['house_area_sqft'] = pd.to_numeric(X['house_area_sqft'], errors='coerce')
test['house_area_sqft'] = pd.to_numeric(test['house_area_sqft'], errors='coerce')

house_area_median = X['house_area_sqft'].median()
X['house_area_sqft'] = X['house_area_sqft'].fillna(house_area_median)
test['house_area_sqft'] = test['house_area_sqft'].fillna(house_area_median)


# For household_size

X['household_size'] = pd.to_numeric(X['household_size'], errors='coerce')
test['household_size'] = pd.to_numeric(test['household_size'], errors='coerce')

household_size_median = X['household_size'].median()
X['household_size'] = X['household_size'].fillna(household_size_median)
test['household_size'] = test['household_size'].fillna(household_size_median)

# round to nearest integer
X['household_size'] = X['household_size'].round()
test['household_size'] = test['household_size'].round()


#Fix garbage values in categorical columns

#replace corrupted heating_type values with 'garbage'
valid_heating = ['gas', 'electric', 'none']
X['heating_type'] = X['heating_type'].where(X['heating_type'].isin(valid_heating), other='garbage')
test['heating_type'] = test['heating_type'].where(test['heating_type'].isin(valid_heating), other='garbage')


cat_features = ['heating_type', 'diet_type']


#one-hot encode
X = pd.get_dummies(X, columns=cat_features)
test = pd.get_dummies(test, columns=cat_features)



#FEATURE ENGINEERING

# List of features that should never be negative but sometimes are
non_negative_cols = [
    'electricity_kwh_per_month',
    'natural_gas_therms_per_month',
    'water_usage_liters_per_day',
    'public_transport_usage_per_week',
    'vehicle_miles_per_month'
]

for col in non_negative_cols:
    X[f'{col}_was_negative'] = (X[col] < 0).astype(int)
    test[f'{col}_was_negative'] = (test[col] < 0).astype(int)
    #median = X[col][X[col] >= 0].median() HOW DOES THIS MAKE THE PREDICTION WORSE
    #X.loc[X[col] < 0, col] = median
    #test.loc[test[col] < 0, col] = median

X['electricity_x_no_solar'] = X['electricity_kwh_per_month'] * (1 - X['uses_solar_panels'])
test['electricity_x_no_solar'] = test['electricity_kwh_per_month'] * (1 - test['uses_solar_panels'])


#X['inefficient_heating'] = -X['home_insulation_quality'] * (1 - X['smart_thermostat_installed'])

#test['inefficient_heating'] = -test['home_insulation_quality'] * (1 - test['smart_thermostat_installed']) #* test['house_area_sqft'] < test['house_area_sqft'].median()


#something relating public transit to vehicle miles?


""" 
X['home_insulation_quality_feature'] = X['home_insulation_quality'] * (X['smart_thermostat_installed']) #soooo close w this one
test['home_insulation_quality_feature'] = test['home_insulation_quality'] * (test['smart_thermostat_installed']) * test['house_area_sqft'] < test['house_area_sqft'].median()


X['high_sqft_x_no_solar'] = X['house_area_sqft'] * (1 - X['uses_solar_panels']) 
test['high_sqft_x_no_solar'] = test['house_area_sqft'] * (1 - test['uses_solar_panels']) 



X['electricity_x_sqft_solar'] = X['electricity_kwh_per_month'] * ( (X['uses_solar_panels']) / X['house_area_sqft']) #if large house, solar will help more
test['electricity_x_sqft_solar'] = test['electricity_kwh_per_month'] * ( (test['uses_solar_panels']) / test['house_area_sqft'])



X['high_electricity_no_solar'] = (
    (X['electricity_kwh_per_month'] > X['electricity_kwh_per_month'].median()) *
    (X['uses_solar_panels'] == 0) * (X['electricity_kwh_per_month'])
).astype(int)
test['high_electricity_no_solar'] = (
    (test['electricity_kwh_per_month'] > test['electricity_kwh_per_month'].median()) *
    (test['uses_solar_panels'] == 0) * (X['electricity_kwh_per_month'])
).astype(int)


X['really_high_electricity_no_solar'] = (
    (X['electricity_kwh_per_month'] > X['electricity_kwh_per_month'].quantile(0.75)) &
    (X['uses_solar_panels'] == 0)
).astype(int)
test['really_high_electricity_no_solar'] = (
    (test['electricity_kwh_per_month'] > test['electricity_kwh_per_month'].quantile(0.75)) &
    (test['uses_solar_panels'] == 0)
).astype(int)


X['rebound'] = (
    (X['electricity_kwh_per_month'] > X['electricity_kwh_per_month'].quantile(0.75)) & (X['energy_efficient_appliances']) #and no solar?
).astype(int)
test['rebound'] = (
    (test['electricity_kwh_per_month'] > test['electricity_kwh_per_month'].quantile(0.75)) & (X['energy_efficient_appliances']) 
).astype(int)


X['really_high_electricity_no_solar'] = (
    (X['electricity_kwh_per_month'] > X['electricity_kwh_per_month'].quartile(75)) &
    (X['uses_solar_panels'] == 0)
) * X['electricity_kwh_per_month']

test['really_high_electricity_no_solar'] = (
    (test['electricity_kwh_per_month'] > test['electricity_kwh_per_month'].quartile(75)) &
    (test['uses_solar_panels'] == 0)
) * test['electricity_kwh_per_month']



X['rebound'] = (
    (X['electricity_kwh_per_month'] > X['electricity_kwh_per_month'].quantile(0.75)) & (X['energy_efficient_appliances']) #and no solar?
) * X['electricity_kwh_per_month']
test['rebound'] = (
    (test['electricity_kwh_per_month'] > test['electricity_kwh_per_month'].quantile(0.75)) & (X['energy_efficient_appliances']) 
) * X['electricity_kwh_per_month']



X['electricity_x_sqft_solar'] = X['electricity_kwh_per_month'] * ( (X['uses_solar_panels']) / X['house_area_sqft']) #if large house, solar will help more
test['electricity_x_sqft_solar'] = test['electricity_kwh_per_month'] * ( (test['uses_solar_panels']) / test['house_area_sqft'])


X['efficient_x_solar'] = X['energy_efficient_appliances'] * (1 - X['uses_solar_panels'])


X['rebound_behavior'] = (X['energy_efficient_appliances'] == 1) & (X['house_area_sqft'] > X['house_area_sqft'].median())
X['electricity_w_rebound'] = X['electricity_kwh_per_month'] * X['rebound_behavior']
test['rebound_behavior'] = (test['energy_efficient_appliances'] == 1) & (test['house_area_sqft'] > X['house_area_sqft'].median())
test['electricity_w_rebound'] = test['electricity_kwh_per_month'] * test['rebound_behavior']


X['uses_heating'] = (X['heating_type_gas'] == 1) | (X['heating_type_electric'] == 1)
X['insulation_x_heating'] = X['home_insulation_quality'] * X['uses_heating']
test['uses_heating'] = (test['heating_type_gas'] == 1) | (test['heating_type_electric'] == 1)
test['insulation_x_heating'] = test['home_insulation_quality'] * test['uses_heating']



X['green_lifestyle'] = (
    (X['recycles_regularly'] == 1) & 
    (X['composts_organic_waste'] == 1) & 
    (X['uses_solar_panels'] == 1)
)
X['elec_but_green'] = X['electricity_kwh_per_month'] * (1 - X['green_lifestyle']) #changed to 1 minus
test['green_lifestyle'] = (
    (test['recycles_regularly'] == 1) & 
    (test['composts_organic_waste'] == 1) & 
    (test['uses_solar_panels'] == 1)
)
test['elec_but_green'] = test['electricity_kwh_per_month'] * (1 - test['green_lifestyle']) #changed to 1 minus


X['plant_based_identity'] = (X['diet_type_vegan'] == 1) | (X['diet_type_vegetarian'] == 1)
X['plant_based_but_eat_meat'] = X['meat_consumption_kg_per_week'] * X['plant_based_identity']
test['plant_based_identity'] = (test['diet_type_vegan'] == 1) | (test['diet_type_vegetarian'] == 1)
test['plant_based_but_eat_meat'] = test['meat_consumption_kg_per_week'] * test['plant_based_identity']


"""



#X['plant_based_diet'] = (X['diet_type_vegan'] == 1) | (X['diet_type_vegetarian'] == 1)
#test['plant_based_diet'] = (test['diet_type_vegan'] == 1) | (test['diet_type_vegetarian'] == 1)

#low home insulation x gas heating
#X['low_insul_x_gas_heat'] = (X['home_insulation_quality'] < X['home_insulation_quality'].median()) * (X['heating_type_gas'] == 1)


#X['solar_x_electric_heating'] = X['uses_solar_panels'] * X['heating_type_electric']
#test['solar_x_electric_heating'] = test['uses_solar_panels'] * test['heating_type_electric']

#X['animal_lover'] = X['owns_pet'] * (1 - X['diet_type_omnivore'])
#test['animal_lover'] = test['owns_pet'] * (1 - test['diet_type_omnivore'])

#energy efficient appliances x house size

#public transit IF you also use a vehicle... could show effort.. idk

#has a pet BUT recycles and composts



#high square footage but have solar panels

#heating type gas and large house size

#smart thermostat and large house

#no energy efficient appliances x large house size

""" 
X['sqft_x_thermo'] = X['house_area_sqft'] * X['smart_thermostat_installed'] * X['home_insulation_quality']
test['sqft_x_thermo'] = test['house_area_sqft'] * test['smart_thermostat_installed'] * test['home_insulation_quality']



X['non_efficient_appliances_x_no_solar'] = (1 - X['energy_efficient_appliances']) * (1 - X['uses_solar_panels'])
test['non_efficient_appliances_x_no_solar'] = (1 - test['energy_efficient_appliances']) * (1 - test['uses_solar_panels'])


X['low_insul_x_gas_x_no_thermo'] = (
    (X['home_insulation_quality'] <= X['home_insulation_quality'].median()) &
    X['heating_type_gas'] &
    (1 - X['smart_thermostat_installed'])
)
test['low_insul_x_gas_x_no_thermo'] = (
    (test['home_insulation_quality'] <= test['home_insulation_quality'].median()) &
    test['heating_type_gas'] &
    (1 - test['smart_thermostat_installed'])
)
"""


#STANDARDIZE NUMERICAL FEATURES

num_features = X.select_dtypes(include=['float64', 'float32', 'int64']).columns.tolist()
binary_like_cols = [col for col in num_features if set(X[col].unique()) <= {0, 1}]
num_features = [col for col in num_features if col not in binary_like_cols]
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
test[num_features] = scaler.transform(test[num_features])

#align columns
X, test = X.align(test, join='left', axis=1, fill_value=0)

#PRINT UDPATED, PREPROCESSED X
print(X.shape)
print(X)


base_models = [
    ("xgb",  xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=3500,
        learning_rate=0.005, #.005
        max_depth=4, #was 4 then 3
        min_child_weight=5,
        subsample=0.8, #was 8 then 6 now 7, now 8 again
        colsample_bytree=0.8,
        gamma = 1.0,
        reg_alpha=5.0,
        reg_lambda=11.0,
        random_state=42,
        n_jobs=-1
    )),
    ("ridge", Ridge(alpha=0.1)),
    ("svr", make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.8)))
]


# Final estimator = conservative XGBoost
final_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=3500,
    learning_rate=0.005, #.005
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.9,
    gamma = 1.0,
    reg_alpha=5.0,  # L1
    reg_lambda=11.0,  # L2
    random_state=42,
    n_jobs=-1,
    verbosity=0
)




stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=final_estimator,
    passthrough=True, #trying this...
    n_jobs=-1
)

print("ensemble model initialized. now training...")


kf = KFold(n_splits=7, shuffle=True, random_state=42)
r2_scores = cross_val_score(stack_model, X, y, cv=kf, scoring="r2")
print("R² scores:", r2_scores)
print(f"Stacking R² (5-Fold CV): {r2_scores.mean():.5f}")

print("Training final model on full dataset...")
# Final model + test preds
stack_model.fit(X, y)
test_preds = stack_model.predict(test)


# Create submission DataFrame
submission = pd.DataFrame({
    'ID': test_ids,
    'carbon_footprint': test_preds
})

# Save to CSV in submission folder
submission.to_csv('../submission/submission.csv', index=False)

print("done")