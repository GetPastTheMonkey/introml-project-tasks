import numpy as np

# Using the scikit-learn library
# https://pypi.org/project/scikit-learn/
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


# Config
folds = 10
lambdas = [0.1, 1, 10, 100, 1000]

# Read CSV data
csv = np.recfromcsv("train.csv")
x = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5, csv.x6, csv.x7, csv.x8, csv.x9, csv.x10))
y = np.column_stack((csv.y, ))

# Compute RMSE using k-fold cross validation. k is determined by the "folds" variable
RMSE = [0 for i in range(len(lambdas))]
fold = KFold(folds)
for train_index, test_index in fold.split(x):
    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    for lambda_index in range(len(lambdas)):
        r = Ridge(lambdas[lambda_index])
        r.fit(x_train, y_train)
        y_predict = r.predict(x_test)
        err = np.sqrt(mean_squared_error(y_test, y_predict))
        RMSE[lambda_index] += err / (1.0 * folds)

# Save resulting RMSE in results.csv
np.savetxt("results.csv", RMSE)
