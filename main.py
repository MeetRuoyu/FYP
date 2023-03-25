import pandas as pd
import numpy as np

import warnings
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
from joblib import dump

from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV

from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

# Reads the Excel sheet, specifying the sheet_name, header and usecols parameters.
file_path = "NOx-data.xlsx"
try:
    df = pd.read_excel(file_path, sheet_name="sheet1", header=0, usecols="A:F")
except FileNotFoundError:
    print("File not found. Please check the file path.")

# Take the first three columns as input and the last three columns as output and convert them to numpy arrays
X = df.iloc[:, 0:3].to_numpy()
y = df.iloc[:, 3:6].to_numpy()

# Divided into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_evaluation(model, params, multi_output=True):
    gs = GridSearchCV(model, params, cv=10)
    if multi_output:
        chain = RegressorChain(gs)
        chain.fit(X_train, y_train)
        y_pred = chain.predict(X_test)
        test_score = chain.score(X_test, y_test)
        output_variables = [f'Variable {i+1}' for i in range(y.shape[1])]
        best_params = [chain.estimators_[i].best_params_ for i in range(y.shape[1])]
        best_scores = [chain.estimators_[i].best_score_ for i in range(y.shape[1])]
    else:
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        test_score = gs.score(X_test, y_test)
        output_variables = ['']
        best_params = [gs.best_params_]
        best_scores = [gs.best_score_]

    results = pd.DataFrame({
        'Output Variable': output_variables,
        'Best parameters': best_params,
        'Best score': best_scores,
        'Test set score': [test_score] * len(output_variables),
        'MAE': [mean_absolute_error(y_test, y_pred)] * len(output_variables),
        'MSE': [mean_squared_error(y_test, y_pred)] * len(output_variables),
        'R2 score': [r2_score(y_test, y_pred)] * len(output_variables)
    })

    return results


# Define the regression models to be trained
models = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    MLPRegressor(max_iter=1000),
    SVR(),
    KNeighborsRegressor(),
    GaussianProcessRegressor(),
    DecisionTreeRegressor(),
    LassoCV(),
]

# Define the hyperparameters to be tuned for each model
params = [
    {},
    {'alpha': [0.1, 1, 10]},
    {'alpha': [0.1, 1, 10]},
    {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
    {'max_depth': [3, 5, 10]},
    {'n_estimators': [50, 100, 200]},
    {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100, 200]},
    {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100, 200]},
    {'hidden_layer_sizes': [(10,), (50,), (100,)]},
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    {'n_neighbors': [3, 5, 10]},
    {},
    {'max_depth': [3, 5, 10]},
    {'eps': [0.001, 0.01, 0.1]},
]
model_names = [
    'LinearRegression()',
    'Ridge()',
    'Lasso()',
    'ElasticNet()',
    'DecisionTreeRegressor()',
    'RandomForestRegressor()',
    'GradientBoostingRegressor()',
    'AdaBoostRegressor()',
    'MLPRegressor(max_iter=1000)',
    'SVR()',
    'KNeighborsRegressor()',
    'GaussianProcessRegressor()',
    'DecisionTreeRegressor()',
    'LassoCV()'
    ]

# Train and evaluate each model
results = []
for i, model in enumerate(models):
    print(f"Training {model_names[i]} ({i + 1}/{len(models)})...")
    results.append(model_evaluation(model, params[i], multi_output=True))

# Concatenate the results of all models into a single DataFrame
all_results = pd.concat(results, ignore_index=True)
print(all_results)


# Random forest regression model
rf_reg = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [5, 7, 9]}
rf_results = model_evaluation(rf_reg, rf_params, multi_output=False)
print("Random Forest:")
print(rf_results.to_markdown(index=False))

# SVR regression model
svr_reg = SVR(kernel='rbf')
svr_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svr_results = model_evaluation(svr_reg, svr_params)
print("SVR:")
print(svr_results.to_markdown(index=False))

# KNN regression model
knn_reg = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_results = model_evaluation(knn_reg, knn_params)
print("KNN:")
print(knn_results.to_markdown(index=False))

# Gaussian Process regression model
gp_reg = GaussianProcessRegressor(random_state=42)
gp_params = {}
gp_results = model_evaluation(gp_reg, gp_params)
print("Gaussian Process:")
print(gp_results.to_markdown(index=False))

# DecisionTreeRegressor regression model
dt_reg = DecisionTreeRegressor(random_state=42)
dt_params = {'max_depth': [3, 5, 10]}
dt_results = model_evaluation(dt_reg, dt_params)
print("DecisionTreeRegressor:")
print(dt_results.to_markdown(index=False))


def model_evaluation_bayes(model, search_space, n_iter=10, multi_output=True, sort_by=None):
    bayes_search = BayesSearchCV(model, search_space, n_iter=n_iter, cv=10, n_jobs=-1)
    if multi_output:
        chain = RegressorChain(bayes_search)
        chain.fit(X_train, y_train)
        y_pred = chain.predict(X_test)
        test_score = chain.score(X_test, y_test)
        output_variables = [f'Variable {i+1}' for i in range(y.shape[1])]
        best_params = [chain.estimators_[i].best_params_ for i in range(y.shape[1])]
        best_scores = [chain.estimators_[i].best_score_ for i in range(y.shape[1])]
    else:
        bayes_search.fit(X_train, y_train)
        y_pred = bayes_search.predict(X_test)
        test_score = bayes_search.score(X_test, y_test)
        output_variables = ['']
        best_params = [bayes_search.best_params_]
        best_scores = [bayes_search.best_score_]

    results = pd.DataFrame({
        'Output Variable': output_variables,
        'Best parameters': best_params,
        'Best score': best_scores,
        'Test set score': [test_score] * len(output_variables),
        'MAE': [mean_absolute_error(y_test, y_pred)] * len(output_variables),
        'MSE': [mean_squared_error(y_test, y_pred)] * len(output_variables),
        'R2 score': [r2_score(y_test, y_pred)] * len(output_variables)
    })

    return results, chain if multi_output else bayes_search


# Random Forest after Bayesian optimization
rf_search_space = {'n_estimators': Integer(10, 100), 'max_depth': Integer(1, 20)}
rf_bayes_results, rf_bayes = model_evaluation_bayes(RandomForestRegressor(random_state=42), rf_search_space, n_iter=100, multi_output=False)
print("Random Forest after Bayesian optimization:")
print(rf_bayes_results.to_markdown(index=False))

# SVR after Bayesian optimization
svr_search_space = {'C': Real(0.1, 20), 'gamma': ['scale', 'auto']}
svr_bayes_results, chain_svr_bayes = model_evaluation_bayes(SVR(kernel='rbf'), svr_search_space, n_iter=100)
print("SVR after Bayesian optimization:")
print(svr_bayes_results.to_markdown(index=False))

# KNN after Bayesian optimization
knn_search_space = {'n_neighbors': Integer(1, 15)}
knn_bayes_results, chain_knn_bayes = model_evaluation_bayes(KNeighborsRegressor(), knn_search_space, n_iter=100)
print("KNN after Bayesian optimization:")
print(knn_bayes_results.to_markdown(index=False))

# Normalisation of input data
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Normalisation of the output data
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)



# Random forest regression model
rf_reg = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [5, 7, 9]}
rf_results = model_evaluation(rf_reg, rf_params, multi_output=False)
print("Random Forest after Normalisation:")
print(rf_results.to_markdown(index=False))

# SVR regression model
svr_reg = SVR(kernel='rbf')
svr_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svr_results = model_evaluation(svr_reg, svr_params)
print("SVR after Normalisation:")
print(svr_results.to_markdown(index=False))

# KNN regression model
knn_reg = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_results = model_evaluation(knn_reg, knn_params)
print("KNN after Normalisation:")
print(knn_results.to_markdown(index=False))

# Gaussian Process regression model
gp_reg = GaussianProcessRegressor(random_state=42)
gp_params = {}
gp_results = model_evaluation(gp_reg, gp_params)
print("Gaussian Process after Normalisation:")
print(gp_results.to_markdown(index=False))

# DecisionTreeRegressor regression model
dt_reg = DecisionTreeRegressor(random_state=42)
dt_params = {'max_depth': [3, 5, 10]}
dt_results = model_evaluation(dt_reg, dt_params)
print("DecisionTreeRegressor after Normalisation:")
print(dt_results.to_markdown(index=False))

# Random Forest after Bayesian optimization
rf_search_space = {'n_estimators': Integer(10, 100), 'max_depth': Integer(1, 20)}
rf_bayes_results, rf_bayes = model_evaluation_bayes(RandomForestRegressor(random_state=42), rf_search_space, n_iter=100, multi_output=False)
print("Random Forest after Bayesian optimization and normalisation:")
print(rf_bayes_results.to_markdown(index=False))

# SVR after Bayesian optimization
svr_search_space = {'C': Real(0.1, 20), 'gamma': ['scale', 'auto']}
svr_bayes_results, chain_svr_bayes = model_evaluation_bayes(SVR(kernel='rbf'), svr_search_space, n_iter=100)
print("SVR after Bayesian optimization and normalisation:")
print(svr_bayes_results.to_markdown(index=False))

# KNN after Bayesian optimization
knn_search_space = {'n_neighbors': Integer(1, 15)}
knn_bayes_results, chain_knn_bayes = model_evaluation_bayes(KNeighborsRegressor(), knn_search_space, n_iter=100)
print("KNN after Bayesian optimization and normalisation:")
print(knn_bayes_results.to_markdown(index=False))



# Creating ensemble models
voting_ensemble_models = []
num_output_variables = y_train.shape[1]

for i in range(num_output_variables):
    single_output_rf = RandomForestRegressor(random_state=42, **rf_bayes.best_params_)
    single_output_svr = SVR(kernel='rbf', **chain_svr_bayes.estimators_[i].best_params_)
    single_output_knn = KNeighborsRegressor(**chain_knn_bayes.estimators_[i].best_params_)
    single_output_gp = gp_reg
    single_output_dt = dt_reg

    ensemble_model = VotingRegressor(
        estimators=[
            ('rf', single_output_rf),
            ('svr', single_output_svr),
            ('knn', single_output_knn),
            ('gp', single_output_gp),
            ('dt', single_output_dt),
        ],
        weights=[4.7, 1.4, 1.3, 1.4, 0.5]
    )

    ensemble_model.fit(X_train, y_train[:, i])
    voting_ensemble_models.append(ensemble_model)

# Make predictions for each output variable
y_pred_normalized = np.zeros_like(y_test)

for i, model in enumerate(voting_ensemble_models):
    y_pred_normalized[:, i] = model.predict(X_test)


# Calculation of assessment indicators
mae = mean_absolute_error(y_test, y_pred_normalized)
mse = mean_squared_error(y_test, y_pred_normalized)
r2 = r2_score(y_test, y_pred_normalized)

# Print assessment indicators
print("Ensemble model results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R2 score: {r2}")






stacking_ensemble_models = []

for i in range(num_output_variables):
    single_output_rf = RandomForestRegressor(random_state=42, **rf_bayes.best_params_)
    single_output_svr = SVR(kernel='rbf', **chain_svr_bayes.estimators_[i].best_params_)
    single_output_knn = KNeighborsRegressor(**chain_knn_bayes.estimators_[i].best_params_)
    single_output_gp = gp_reg
    single_output_dt = dt_reg

    ensemble_model = StackingRegressor(
        estimators=[
            ('svr', single_output_svr),
            ('knn', single_output_knn),
            ('rf', single_output_rf),
            ('gp', single_output_gp),
            ('dt', single_output_dt)
        ],
        final_estimator=single_output_rf
    )

    ensemble_model.fit(X_train, y_train[:, i])
    stacking_ensemble_models.append(ensemble_model)

y_pred_normalized = np.zeros_like(y_test)

for i, model in enumerate(stacking_ensemble_models):
    y_pred_normalized[:, i] = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_normalized)
mse = mean_squared_error(y_test, y_pred_normalized)
r2 = r2_score(y_test, y_pred_normalized)

print("Ensemble model results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R2 score: {r2}")


dump(scaler_x, "scaler_x.joblib")
dump(scaler_y, "scaler_y.joblib")

dump(rf_reg, "Random Forest.joblib")
dump(svr_reg, "SVR.joblib")
dump(knn_reg, "KNN.joblib")
dump(gp_reg, "GaussianProcessRegressor.joblib")
dump(dt_reg, "DecisionTreeRegressor.joblib")

dump(rf_bayes, "Random Forest Bayesian optimization.joblib")
dump(chain_svr_bayes, "SVR Bayesian optimization.joblib")
dump(chain_knn_bayes, "KNN Bayesian optimization.joblib")

dump(voting_ensemble_models, "VotingRegressor Ensemble.joblib")
dump(stacking_ensemble_models, "StackingRegressor Ensemble.joblib")





def get_prediction():
    input_values = [float(input1.get()), float(input2.get()), float(input3.get())]
    input_array = np.array(input_values).reshape(1, -1)

    normalized_input_array = scaler_x.transform(input_array)

    if model_var.get() == 'Random Forest Bayesian optimization':
        prediction = rf_bayes.predict(input_array)
    elif model_var.get() == 'SVR Bayesian optimization':
        prediction = chain_svr_bayes.predict(input_array)
    elif model_var.get() == 'KNN Bayesian optimization':
        prediction = chain_knn_bayes.predict(input_array)
    elif model_var.get() == 'GaussianProcessRegressor':
        prediction = gp_results.predict(input_array)
    elif model_var.get() == 'DecisionTreeRegressor':
        prediction = dt_results.predict(input_array)
    elif model_var.get() == 'VotingRegressor Ensemble':
        prediction = np.zeros_like(normalized_input_array)
        for i, model in enumerate(voting_ensemble_models):
            prediction[:, i] = model.predict(normalized_input_array)
    elif model_var.get() == 'StackingRegressor Ensemble':
        prediction = np.zeros_like(normalized_input_array)
        for i, model in enumerate(stacking_ensemble_models):
            prediction[:, i] = model.predict(normalized_input_array)

    prediction = scaler_y.inverse_transform(prediction)

    result_label.config(
        text=f'Nox Concentration (ppm): {prediction[0][0]:.2f}\nEC (MJ/mol): {prediction[0][1]:.2f}\nSelectivity (NO) (%): {prediction[0][2]:.2f}')

root = tk.Tk()
root.title("Model Predictor")
root.geometry("300x300")

model_var = tk.StringVar(root)
model_var.set("Random Forest Bayesian optimization")

model_label = tk.Label(root, text="Select model:")
model_label.grid(row=0, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

model_dropdown = ttk.Combobox(root, textvariable=model_var, values=['Random Forest Bayesian optimization', 'SVR Bayesian optimization', 'KNN Bayesian optimization', 'GaussianProcessRegressor', 'DecisionTreeRegressor', 'VotingRegressor Ensemble', 'StackingRegressor Ensemble'])
model_dropdown.grid(row=0, column=1, padx=(0, 10), pady=(10, 0), sticky="e")

input1_label = tk.Label(root, text="P(W):")
input1_label.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

input1 = tk.Entry(root)
input1.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="e")

input2_label = tk.Label(root, text="N2/O2 ratio:")
input2_label.grid(row=2, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

input2 = tk.Entry(root)
input2.grid(row=2, column=1, padx=(0, 10), pady=(10, 0), sticky="e")

input3_label = tk.Label(root, text="Flow Rate (slm):")
input3_label.grid(row=3, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

input3 = tk.Entry(root)
input3.grid(row=3, column=1, padx=(0, 10), pady=(10, 0), sticky="e")

submit_button = tk.Button(root, text="Predict", command=get_prediction)
submit_button.grid(row=4, column=1, padx=(0, 10), pady=(10, 0), sticky="e")

result_label = tk.Label(root, text="")
result_label.grid(row=5, column=0, columnspan=2, padx=(10, 10), pady=(10, 0))

root.mainloop()
