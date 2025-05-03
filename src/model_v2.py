
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import StackingRegressor, RandomForestRegressor
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import optuna

print("Model script starting...")

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

#For house_area_sqft

X['house_area_sqft'] = pd.to_numeric(X['house_area_sqft'], errors='coerce')
test['house_area_sqft'] = pd.to_numeric(test['house_area_sqft'], errors='coerce')

house_area_median = X['house_area_sqft'].median()
X['house_area_sqft'] = X['house_area_sqft'].fillna(house_area_median)
test['house_area_sqft'] = test['house_area_sqft'].fillna(house_area_median)


#For household_size

X['household_size'] = pd.to_numeric(X['household_size'], errors='coerce')
test['household_size'] = pd.to_numeric(test['household_size'], errors='coerce')

household_size_median = X['household_size'].median()
X['household_size'] = X['household_size'].fillna(household_size_median)
test['household_size'] = test['household_size'].fillna(household_size_median)

#round to nearest integer
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

#List of features that should never be negative but sometimes are
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
    #median = X[col][X[col] >= 0].median() 
    #X.loc[X[col] < 0, col] = median
    #test.loc[test[col] < 0, col] = median


X['electricity_x_no_solar'] = X['electricity_kwh_per_month'] * (1 - X['uses_solar_panels'])
test['electricity_x_no_solar'] = test['electricity_kwh_per_month'] * (1 - test['uses_solar_panels'])



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



#MODEL STUFF


#HYPERPARAM OPTIMIZATION

def xgb_objective(trial):

    params = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 3e-3, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 2900, 3400),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 9),
        'gamma': trial.suggest_float('gamma', 1e-3, 1, log=True),
        'subsample': trial.suggest_float('subsample', 0.75, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 25, log=True)
    }

    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    xgb_scores = []
    

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        xgb_preds = xgb_model.predict(X_val_fold)
        score = r2_score(y_val_fold, xgb_preds)
        xgb_scores.append(r2_score(y_val_fold, xgb_preds))

        print(f"Trial {trial.number} | Fold {fold+1} R²: {score:.4f}")


    return np.mean(xgb_scores)

""" 
study = optuna.create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
print("Best trial:")
print(study.best_trial.params)
print(f"Best R²: {study.best_value:.4f}")
"""


def ridge_objective(trial):
    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    ridge_scores = []


    alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        ridge_model = Ridge(alpha=alpha) 
        ridge_model.fit(X_train, y_train)
        preds = ridge_model.predict(X_val)
        score = r2_score(y_val, preds)
        ridge_scores.append(score)

        print(f"Trial {trial.number} | Fold {fold+1} R²: {score:.4f}")

    return np.mean(ridge_scores)
""" 
study = optuna.create_study(direction='maximize')
study.optimize(ridge_objective, n_trials=50, timeout=600, show_progress_bar=True)
print("Best trial:")
print(study.best_trial.params)
print(f"Best R²: {study.best_value:.4f}")
"""


def svr_objective(trial):
    C = trial.suggest_float('C', 40, 65, log=True)
    epsilon = trial.suggest_float('epsilon', 0.05, 0.15)
    kernel = 'rbf' #trial.suggest_categorical('kernel', ['linear', 'rbf'])

    params = {
        'C': C,
        'epsilon': epsilon,
        'kernel': kernel
    }

    if kernel == 'rbf':
        params['gamma'] = trial.suggest_float('gamma', 0.01, 0.08, log=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    svr_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        svr_model = SVR(**params)
        svr_model.fit(X_train, y_train)
        preds = svr_model.predict(X_val)
        
        score = r2_score(y_val, preds)
        svr_scores.append(score)

        print(f"Trial {trial.number} | Fold {fold+1} R²: {score:.4f}")

    return np.mean(svr_scores)

""" 
study = optuna.create_study(direction='maximize')
study.optimize(svr_objective, n_trials=50, timeout=3600, show_progress_bar=True)
print("Best trial:")
print(study.best_trial.params)
print(f"Best R²: {study.best_value:.4f}")
"""


def initialize_models():
    xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=3160, 
            learning_rate=0.013747631749763938, 
            early_stopping_rounds=50,
            eval_metric='rmse',
            max_depth=3,
            min_child_weight=6,
            subsample=0.7665642854389441, 
            colsample_bytree=0.7993846232206838, 
            gamma=0.0031267985476264135,
            reg_alpha=3.60493467222669, 
            reg_lambda=16.75634856112939, 
            random_state=42,
            n_jobs=-1
        )

    ridge_model = Ridge(alpha=0.3462756875931813)

    svr_model = SVR(
        C=63.65623751727156, # 51.60643974838364, 43.285903986533 
        epsilon=0.11192645956514154, # 0.1301830896767313, 0.05131070202415048 
        kernel='rbf',
        gamma=0.03969394679777197 #bc used rbf
    )
    return xgb_model, ridge_model, svr_model


#generate OOF predictions from base models
print("Generating OOF predictions from base models (XGB, Ridge, SVR)...")

oof_preds = np.zeros((X.shape[0], 3))
kf = KFold(n_splits=7, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

   
    xgb_fold, ridge_fold, svr_fold = initialize_models()

    xgb_fold.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    ridge_fold.fit(X_train, y_train)
    svr_fold.fit(X_train, y_train)

    oof_preds[val_idx, 0] = xgb_fold.predict(X_val)
    oof_preds[val_idx, 1] = ridge_fold.predict(X_val)
    oof_preds[val_idx, 2] = svr_fold.predict(X_val)

    print(f"Fold {fold+1} — XGB, Ridge, SVR fitted and predicted.")

    xgb_r2 = r2_score(y_val, xgb_fold.predict(X_val))
    ridge_r2 = r2_score(y_val, ridge_fold.predict(X_val))
    svr_r2 = r2_score(y_val, svr_fold.predict(X_val))

    print(f"Fold {fold+1} R²s — XGB: {xgb_r2:.4f}, Ridge: {ridge_r2:.4f}, SVR: {svr_r2:.4f}")
        

""" 
#Tune ridge meta-model on OOF predictions w optuna
def ridge_meta_model_objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 10)

    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(oof_preds)):
        X_train, X_val = oof_preds[train_idx], oof_preds[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        scores.append(score)

        print(f"Trial {trial.number} | Fold {fold+1} R²: {score:.4f}")

    return np.mean(scores)

#Tune xgboost meta model on OOF predictions w optuna
def xgb_meta_model_objective(trial):
    print("optimizing meta model hyperparams")
    params = {
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 2.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
    }


    kf = KFold(n_splits=7, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(oof_preds)):
        X_train, X_val = oof_preds[train_idx], oof_preds[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            #early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        scores.append(score)

        print(f"Trial {trial.number} | Fold {fold+1} R²: {score:.4f}")

    return np.mean(scores)

study = optuna.create_study(direction='maximize')
#study.optimize(ridge_meta_model_objective, timeout=120, show_progress_bar=True)
study.optimize(xgb_meta_model_objective, n_trials=200, show_progress_bar=True)
print("Best meta-model R²: ", study.best_value)
print("Best params: ", study.best_trial.params)


"""

ridge_meta_model = Ridge(alpha=9.999966467224843)
print("meta model initialized! now training...")
ridge_meta_model.fit(oof_preds, y)

""" 
xgb_meta_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=3160, 
            learning_rate=0.013747631749763938,
            max_depth=3,
            min_child_weight=6,
            subsample=0.7665642854389441, 
            colsample_bytree=0.7993846232206838, 
            gamma=0.0031267985476264135,
            reg_alpha=3.60493467222669, 
            reg_lambda=16.75634856112939, 
            random_state=42,
            n_jobs=-1
        )


xgb_meta_model.fit(oof_preds, y,
                   verbose=False
                   )

"""
print("Meta-model R² on OOF:", r2_score(y, ridge_meta_model.predict(oof_preds)))






#FITTING ON FULL TRAINING SET

#INITIALIZE MODELS
xgb_model, ridge_model, svr_model = initialize_models()

print("Training final base models on full dataset...")

#FIT BASE MODELS ON FULL TRAINING SET
xgb_model.fit(
        X, y,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
ridge_model.fit(X, y)
svr_model.fit(X, y)


# USE FULL-DATA-TRAINED BASE MODELS TO GENERATE PREDICTIONS FOR META MODEL PARAMS
train_preds_full = np.column_stack([
    xgb_model.predict(X),
    ridge_model.predict(X),
    svr_model.predict(X)
])

# TRAIN META MODEL ON THE FULL-TRAINING BASE MODEL PREDICTION
print("final meta model initialized. now training...")
ridge_meta_model.fit(train_preds_full, y)


print("Everything trained! Generating test set predictions...")

#GENERATE PREDICTIONS FOR EACH BASE MODEL
base_test_preds = np.column_stack([
    xgb_model.predict(test),
    ridge_model.predict(test),
    svr_model.predict(test)
])

#GENERATE FINAL PREDICTIONS USING META MODEL
final_preds = ridge_meta_model.predict(base_test_preds)



#create submission df
submission = pd.DataFrame({
    'ID': test_ids,
    'carbon_footprint': final_preds
})

#save to CSV in submission folder
submission.to_csv('../submission/submission.csv', index=False)


print("done")