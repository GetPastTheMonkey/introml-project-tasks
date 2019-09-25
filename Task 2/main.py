import numpy as np

# Using the scikit-learn library
# https://pypi.org/project/scikit-learn/
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

# Config
c_max = 5
c_step = 0.1
gamma_max = 0.1
gamma_step = 0.01
folds = 50
cache_size = 4000
selector_range = range(7, 13)

# Load training data
csv = np.recfromcsv("train.csv")
x_train = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5, csv.x6, csv.x7, csv.x8, csv.x9, csv.x10, csv.x11,
                           csv.x12, csv.x13, csv.x14, csv.x15, csv.x16, csv.x17, csv.x18, csv.x19, csv.x20))
y_train = np.column_stack((csv.y,))

# Load testing data
csv = np.recfromcsv("test.csv")
x_test = np.column_stack((csv.x1, csv.x2, csv.x3, csv.x4, csv.x5, csv.x6, csv.x7, csv.x8, csv.x9, csv.x10, csv.x11,
                          csv.x12, csv.x13, csv.x14, csv.x15, csv.x16, csv.x17, csv.x18, csv.x19, csv.x20))

# Scale training and testing X
# x_train = scale(x_train)
# x_test = scale(x_test)

# Do k-fold
best_score = None
best_svc = None
best_c = None
best_gamma = None
best_selector = None
best_selector_k = None
fold = KFold(folds)
i = 0
for train_index, test_index in fold.split(x_train):
    x_train_train = x_train[train_index]
    y_train_train = y_train[train_index]
    x_train_test = x_train[test_index]
    y_train_test = y_train[test_index]

    for c_iter in np.arange(c_step, c_max, c_step):
        for gamma_iter in np.arange(gamma_step, gamma_max, gamma_step):
            for sel in selector_range:
                # Selector
                selector = SelectKBest(k=sel)
                selector.fit(x_train_train, y_train_train)
                x_transformed_train = selector.transform(x_train_train)
                x_transformed_test = selector.transform(x_train_test)

                # SVC
                svc = SVC(C=c_iter, cache_size=cache_size, probability=False, gamma=gamma_iter,
                          decision_function_shape="ovo")
                svc.fit(x_transformed_train, y_train_train)
                y_train_predict = svc.predict(x_transformed_test)

                score = accuracy_score(y_train_test, y_train_predict)
                if best_score is None or score > best_score:
                    best_score = score
                    best_svc = svc
                    best_c = c_iter
                    best_gamma = gamma_iter
                    best_selector = selector
                    best_selector_k = sel
                    print "New best score: {}".format(best_score)

                print "\tFinished fold={}, c={}, gamma={}, selector={}".format(i, c_iter, gamma_iter, sel)

                if best_score == 1.0:
                    break
            if best_score == 1.0:
                break
        if best_score == 1.0:
            break
    if best_score == 1.0:
        break

    i += 1

# Predict data with best selector and best SVC
x_test_transformed = best_selector.transform(x_test)
y_predict = best_svc.predict(x_test_transformed)

# Save to file
with open("result.csv", "w") as f:
    f.write("Id,y\n")
    _id = 2000
    for y in y_predict:
        f.write("{},{}\n".format(_id, y))
        _id += 1

print "Finished with best_score={}, best_c={}, best_gamma={}, best_selector_k={}".format(best_score, best_c, best_gamma,
                                                                                         best_selector_k)
