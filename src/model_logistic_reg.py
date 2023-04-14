

## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Importing py files
# exec(open(f'{SRC_PATH}/scrapper.py').read())
exec(open(f'{SRC_PATH}/csv_import.py').read())

## IMPORTING THE TRAINING AND TEST DATA

# parameters
# train_version = list(range(24, 35))
# test_version = [35, 36]
# jump_threshold = 0.05

# import
X_train, Y_train = import_several_csv(train_version, to_np = False)
X_test, Y_test = import_several_csv(test_version, to_np = False)

Y_test_ = Y_test.copy().drop_duplicates()
# preprocess
X_train, Y_train = preprocess_df(X_train, Y_train, jump_threshold, True, NORM)
X_test, Y_test = preprocess_df(X_test, Y_test, jump_threshold, True, NORM)
##

def plot(Y_pred, title:str, close=False) :
    plt.plot(Y_pred, "o", label = f"{title}", markersize=8)
    top = np.mean(Y_test_) + np.std(Y_test_)
    bot = np.mean(Y_test_) - np.std(Y_test_)

    for i in list(np.where(Y_test_ > top)[0]) :
        plt.axvspan(i-.5, i+.5, 0.75, 1, facecolor='limegreen', alpha=0.5)

    for i in set(list(np.where(Y_test_ > bot))[0]) & set(list(np.where(Y_test_ < top))[0]) :
        plt.axvspan(i-.5, i+.5,0.25, 0.75, facecolor='gold', alpha=0.5)

    for i in list(np.where(Y_test_ < bot)[0]) :
        plt.axvspan(i-.5, i+.5, 0, 0.25, facecolor='salmon', alpha=0.5)

    plt.title(f"{title}\nVersion {train_version}")
    plt.grid()
    plt.legend()
    plt.savefig(f'{tmp_path}/photos/{title}.png')

    if not close :
        plt.show()
    plt.close()

## LOGISTIC REGRESSION

# p = 0.1

logisticRegr = LogisticRegression(solver= "liblinear", penalty="l1",
                                    multi_class = "ovr",
                                    C = .3,
                                    fit_intercept= False, max_iter = 500,
                                    class_weight = "balanced" )
                                    # class_weight = {0:p, 1:1-p})

# Fit and predict
logisticRegr.fit(X_train, Y_train.flatten())
Y_pred = logisticRegr.predict_proba(X_test)[:,1]

score = logisticRegr.score(X_test, Y_test)
print(score)

plot(Y_pred, "Logistic regression", True)
pd.DataFrame(Y_pred).to_csv(f"{RES_PATH}/logistic_reg.csv", index = True)

## XGB

from xgboost import XGBClassifier

m = XGBClassifier(
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)

# Fit and predict
m.fit(X_train, Y_train.flatten())
Y_pred = m.predict_proba(X_test)[:,1]

plot(Y_pred, "XGB classification", True)
pd.DataFrame(Y_pred).to_csv(f"{RES_PATH}/XGB.csv", index = True)

## ISOLATION FOREST
from sklearn.ensemble import IsolationForest



cont = len(Y_test[Y_test == 1]) / len(Y_test)

# isf = IsolationForest(n_jobs=-1, random_state=1)
isf = IsolationForest(n_estimators = 500,
                random_state = 0, contamination = round(cont, 1) ,
                bootstrap = False)

# prediction
# isf.fit(X_train)
# Y_pred = isf.predict(X_test)

Y_pred = isf.fit_predict(X_test)

#process
Y_pred = -Y_pred # -1 are outliers.
Y_pred[Y_pred==-1] = 0

# Plot

plot(Y_pred, "Isolation forest", True)
pd.DataFrame(Y_pred).to_csv(f"{RES_PATH}/isolation_forest.csv", index = True)


##
