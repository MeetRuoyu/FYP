import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tkinter as tk

from tkinter import ttk

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingRegressor
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

# # 使用StandardScaler对数据进行归一化处理
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

# Divided into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random forest regression model
rf_reg = RandomForestRegressor(random_state=42)

# Defining the parameter space
rf_params = {'n_estimators': [10, 50, 100],
             'max_depth': [5, 7, 9]}

# Conduct a grid search, using 10-fold cross-validation
rf_gs = GridSearchCV(rf_reg, rf_params, cv=10)

# Fit the multi-output random forest regressor
rf_gs.fit(X_train, y_train)

# Evaluate the multi-output random forest regressor
y_pred_rf = rf_gs.predict(X_test)

# Output model testing results
rf_results = pd.DataFrame({
    'Best parameters': [rf_gs.best_params_],
    'Best score': [rf_gs.best_score_],
    'Test set score': [rf_gs.score(X_test, y_test)],
    'MAE': [mean_absolute_error(y_test, y_pred_rf)],
    'MSE': [mean_squared_error(y_test, y_pred_rf)],
    'R2 score': [r2_score(y_test, y_pred_rf)]
})
print("Random Forest:")
print(rf_results.to_markdown(index=False))



# SVR regression model
svr_reg = SVR(kernel='rbf')

# Defining the parameter space
svr_params = {'C': [0.1, 1, 10],
              'gamma': ['scale', 'auto']}

# Conduct a grid search, using 10-fold cross-validation
svr_gs = GridSearchCV(svr_reg, svr_params, cv=10)

# Wrapping the SVR model in a RegressorChain
svr_chain = RegressorChain(svr_gs)

# Fitting the RegressorChain model
svr_chain.fit(X_train, y_train)

# Evaluate the RegressorChain model
y_pred_svr = svr_chain.predict(X_test)

# Output model testing results
svr_results = pd.DataFrame({
    'Output Variable': [f'Variable {i+1}' for i in range(y.shape[1])],
    'Best parameters': [svr_chain.estimators_[i].best_params_ for i in range(y.shape[1])],
    'Best score': [svr_chain.estimators_[i].best_score_ for i in range(y.shape[1])]
})

svr_results2 = pd.DataFrame({
    'MAE': [mean_absolute_error(y_test, y_pred_svr)],
    'MSE': [mean_squared_error(y_test, y_pred_svr)],
    'R2 score': [r2_score(y_test, y_pred_svr)]
})

svr_results['Test set score'] = svr_chain.score(X_test, y_test)
print("SVR:")
print(svr_results.to_markdown(index=False))
print(svr_results2.to_markdown(index=False))





# KNN regression model
knn_reg = KNeighborsRegressor()

# Defining the parameter space
knn_params = {'n_neighbors': [3, 5, 7, 9]}

# Conduct a grid search, using 10-fold cross-validation
knn_gs = GridSearchCV(knn_reg, knn_params, cv=10)

# Wrap the random forest regressor with MultiOutputRegressor
knn_chain = RegressorChain(knn_gs)

# Fit the multi-output random forest regressor
knn_chain.fit(X_train, y_train)

# Evaluate the multi-output random forest regressor
y_pred_knn = knn_chain.predict(X_test)

# Output model testing results
knn_results = pd.DataFrame({
    'Output Variable': [f'Variable {i+1}' for i in range(y.shape[1])],
    'Best parameters': [knn_chain.estimators_[i].best_params_ for i in range(y.shape[1])],
    'Best score': [knn_chain.estimators_[i].best_score_ for i in range(y.shape[1])]
})

knn_results2 = pd.DataFrame({
    'MAE': [mean_absolute_error(y_test, y_pred_knn)],
    'MSE': [mean_squared_error(y_test, y_pred_knn)],
    'R2 score': [r2_score(y_test, y_pred_knn)]
})

knn_results['Test set score'] = knn_chain.score(X_test, y_test)
print("KNN:")
print(knn_results.to_markdown(index=False))
print(knn_results2.to_markdown(index=False))





# Define search spaces
rf_search_space = {'n_estimators': Integer(10, 100),
                   'max_depth': Integer(1, 20)}
rf_bayes = BayesSearchCV(RandomForestRegressor(random_state=42), rf_search_space, n_iter=100, cv=10, n_jobs=-1)
rf_bayes.fit(X_train, y_train)
rf_y_pred = rf_bayes.predict(X_test)

# Output model testing results
rf_bayes_results = pd.DataFrame({
    'Best parameters': [rf_bayes.best_params_],
    'Best score': [rf_bayes.best_score_],
    'Test set score': [rf_bayes.score(X_test, y_test)],
    'MAE': [mean_absolute_error(y_test, rf_y_pred)],
    'MSE': [mean_squared_error(y_test, rf_y_pred)],
    'R2 score': [r2_score(y_test, rf_y_pred)]
})
print("Random Forest after Bayesian optimization:")
print(rf_bayes_results.to_markdown(index=False))

# Ranking the importance of parameters
# Get feature importance
importances = rf_bayes.best_estimator_.feature_importances_

# Ranking according to feature importance
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

# Converting X back to a Pandas DataFrame
X = df.iloc[:, :-3]

# Print ranking
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))



# Define search spaces
svr_search_space = {'C': Real(0.1, 20),
                    'gamma': ['scale', 'auto']}
svr_bayes = BayesSearchCV(SVR(kernel='rbf'), svr_search_space, n_iter=10, cv=10, n_jobs=-1)

# Use RegressorChain to wrap SVRs into multi-output regressors
chain_svr_bayes = RegressorChain(svr_bayes)
chain_svr_bayes.fit(X_train, y_train)
chain_svr_y_pred = chain_svr_bayes.predict(X_test)

# Output model testing results
svr_bayes_results = pd.DataFrame({
    'Output Variable': [f'Variable {i+1}' for i in range(y.shape[1])],
    'Best parameters': [chain_svr_bayes.estimators_[i].best_params_ for i in range(y.shape[1])],
    'Best score': [chain_svr_bayes.estimators_[i].best_score_ for i in range(y.shape[1])]
})

svr_bayes_results2 = pd.DataFrame({
    'MAE': [mean_absolute_error(y_test, chain_svr_y_pred)],
    'MSE': [mean_squared_error(y_test, chain_svr_y_pred)],
    'R2 score': [r2_score(y_test, chain_svr_y_pred)]
})

svr_bayes_results['Test set score'] = chain_svr_bayes.score(X_test, y_test)
print("SVR after Bayesian optimization::")
print(svr_bayes_results.to_markdown(index=False))
print(svr_bayes_results2.to_markdown(index=False))



knn_search_space = {'n_neighbors': Integer(1, 15)}
knn_bayes = BayesSearchCV(KNeighborsRegressor(), knn_search_space, n_iter=10, cv=10, n_jobs=-1)


# 使用RegressorChain将KNN包装成多输出回归器
chain_knn_bayes = RegressorChain(knn_bayes)
chain_knn_bayes.fit(X_train, y_train)
chain_knn_y_pred = chain_knn_bayes.predict(X_test)

# Output model testing results
knn_bayes_results = pd.DataFrame({
    'Output Variable': [f'Variable {i+1}' for i in range(y.shape[1])],
    'Best parameters': [chain_knn_bayes.estimators_[i].best_params_ for i in range(y.shape[1])],
    'Best score': [chain_knn_bayes.estimators_[i].best_score_ for i in range(y.shape[1])]
})

knn_bayes_results2 = pd.DataFrame({
    'MAE': [mean_absolute_error(y_test, chain_knn_y_pred)],
    'MSE': [mean_squared_error(y_test, chain_knn_y_pred)],
    'R2 score': [r2_score(y_test, chain_knn_y_pred)]
})

knn_bayes_results['Test set score'] = chain_knn_bayes.score(X_test, y_test)
print("KNN after Bayesian optimization:")
print(knn_bayes_results.to_markdown(index=False))
print(knn_bayes_results2.to_markdown(index=False))


# Drawing graphical interfaces, get user input and make predictions
def get_prediction():
    input_values = [float(input1.get()), float(input2.get()), float(input3.get())]
    input_array = np.array(input_values).reshape(1, -1)

    if model_var.get() == 'Random Forest':
        prediction = rf_gs.predict(input_array)
    elif model_var.get() == 'SVR':
        prediction = svr_chain.predict(input_array)
    elif model_var.get() == 'KNN':
        prediction = knn_chain.predict(input_array)
    elif model_var.get() == 'Random Forest Bayesian optimization':
        prediction = rf_bayes.predict(input_array)
    elif model_var.get() == 'SVR Bayesian optimization':
        prediction = chain_svr_bayes.predict(input_array)
    elif model_var.get() == 'KNN Bayesian optimization':
        prediction = chain_knn_bayes.predict(input_array)
    #prediction_original_scale = scaler.inverse_transform(prediction)
    result_label.config(
#        text=f'Nox Concentration (ppm): {prediction_original_scale[0][0]:.2f}\nEC (MJ/mol): {prediction_original_scale[0][1]:.2f}\nSelectivity (NO) (%): {prediction_original_scale[0][2]:.2f}')
        text = f'Nox Concentration (ppm): {prediction[0][0]:.2f}\nEC (MJ/mol): {prediction[0][1]:.2f}\nSelectivity (NO) (%): {prediction[0][2]:.2f}')
# Create a tkinter window and set the window title
root = tk.Tk()
root.title("Model Predictor")

# Create drop-down box for model selection
model_var = tk.StringVar(root)
model_var.set("Random Forest")

model_dropdown = ttk.Combobox(root, textvariable=model_var, values=['Random Forest', 'SVR', 'KNN', 'Random Forest Bayesian optimization', 'SVR Bayesian optimization', 'KNN Bayesian optimization'])
model_dropdown.grid(row=0, column=1)

model_label = tk.Label(root, text="Select model:")
model_label.grid(row=0, column=0)

input1 = tk.Entry(root)
input1.grid(row=1, column=1)

input1_label = tk.Label(root, text="P(W):")
input1_label.grid(row=1, column=0)

input2 = tk.Entry(root)
input2.grid(row=2, column=1)

input2_label = tk.Label(root, text="N2/O2 ratio")
input2_label.grid(row=2, column=0)

input3 = tk.Entry(root)
input3.grid(row=3, column=1)

input3_label = tk.Label(root, text="Flow Rate (slm)")
input3_label.grid(row=3, column=0)

submit_button = tk.Button(root, text="Predict", command=get_prediction)
submit_button.grid(row=4, column=1)

result_label = tk.Label(root, text="")
result_label.grid(row=5, column=1)

root.mainloop()





rf_train_sizes, rf_train_scores, rf_test_scores = learning_curve(rf_bayes.best_estimator_, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure()
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.grid()
plt.plot(rf_train_sizes, np.mean(rf_train_scores, axis=1), 'o-', color="r", label="Training score")
plt.plot(rf_train_sizes, np.mean(rf_test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()



def plot_learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, validation_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=cv,
                                                                  scoring=scoring)

    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Error')
    plt.plot(train_sizes, validation_scores_mean, 'o-', color='g', label='Validation Error')

    plt.xlabel('Training set size')
    plt.ylabel('Mean Squared Error')
    plt.ylim(0.0, 1.1)
    plt.grid()
    plt.legend()
    plt.title('Learning Curve')
    plt.show()


# Use the Random Forest model from GridSearchCV
svr_bayes_best_model = chain_svr_bayes.estimators_[0].best_estimator_

# Plot learning curve
plot_learning_curve(svr_bayes_best_model, X_train, y_train)


# Use the Random Forest model from GridSearchCV
knn_bayes_best_model = chain_knn_bayes.estimators_[0].best_estimator_

# Plot learning curve
plot_learning_curve(knn_bayes_best_model, X_train, y_train)






file_path = "data3.xlsx"
try:
    df1 = pd.read_excel(file_path, sheet_name="sheet1", header=0, usecols="A:C")
except FileNotFoundError:
    print("File not found. Please check the file path.")

X1 = df1.iloc[:, 0:3].to_numpy()
# Xtest_norm = scaler.transform(X1)

# Make predictions using the trained model
# y_pred_norm = rf_bayes.predict(Xtest_norm)
# y_pred = scaler.inverse_transform(y_pred_norm)
y_pred = rf_bayes.predict(X1)


print(y_pred)

# Save the predicted values to the original dataframe
df1["Prediction1"] = y_pred[:, 0]
df1["Prediction2"] = y_pred[:, 1]
df1["Prediction3"] = y_pred[:, 2]

# Save the dataframe to an Excel file
df1.to_excel("data1_predictions3.xlsx", index=False)
