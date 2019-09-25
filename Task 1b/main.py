import numpy as np

# Using the scikit-learn library
# https://pypi.org/project/scikit-learn/
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Config
folds = 21
lam = 0.65
ratio = 1.35

# Read CSV data
csv = np.recfromcsv("train.csv")
x = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5))
y = np.column_stack((csv.y, ))


# Define transformation function
def transform(x_list):
    return [
        x_list[0],
        x_list[1],
        x_list[2],
        x_list[3],
        x_list[4],
        x_list[0]**2,
        x_list[1]**2,
        x_list[2]**2,
        x_list[3]**2,
        x_list[4]**2,
        np.exp(x_list[0]),
        np.exp(x_list[1]),
        np.exp(x_list[2]),
        np.exp(x_list[3]),
        np.exp(x_list[4]),
        np.cos(x_list[0]),
        np.cos(x_list[1]),
        np.cos(x_list[2]),
        np.cos(x_list[3]),
        np.cos(x_list[4]),
        1
    ]


# Build feature matrix
features = np.apply_along_axis(transform, axis=1, arr=x)

# Folds
fold = KFold(folds)
largest_err = None
weights = None
best_lambda = None
i = 0

for train_index, test_index in fold.split(features):
    x_train = features[train_index]
    x_test = features[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    r = ElasticNet(alpha=lam, l1_ratio=ratio, fit_intercept=False)
    r.fit(x_train, y_train)
    y_predict = r.predict(X=x_test)
    err = np.sqrt(mean_squared_error(y_test, y_predict))
    if largest_err is None or err > largest_err:
        largest_err = err
        weights = r.coef_
        best_lambda = lam
    print "Finished i={}".format(i)
    i += 1


print "Err:\t{}\nAlpha:\t{}".format(largest_err, best_lambda)
np.savetxt("results.csv", weights, delimiter="\n")
