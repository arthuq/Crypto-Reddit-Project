
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
# train_version = list(range(23, 35))
# test_version = [35, 36]
# jump_threshold = 0.05

# import
X_train, Y_train = import_several_csv(train_version, to_np = True)
X_test, Y_test = import_several_csv(test_version, to_np = True)

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


## ENCODING DATA
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown = 'ignore')

# Encode data
# X_train = encoder.fit_transform(X_train)
# X_test = encoder.transform(X_test)

## TRAINING TREE

from sklearn.tree import DecisionTreeClassifier

# clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

clf_tree = DecisionTreeClassifier(criterion = "gini",
                                    random_state=0,
                                    max_depth=4,
                                    min_samples_split = 25,
                                    min_samples_leaf = 5,
                                    max_features = 4)

# Fit and predict
clf_tree.fit(X_train, Y_train)

Y_pred = clf_tree.predict(X_test)
# Y_pred = clf_tree.predict_proba(X_test)
print(Y_pred)

clf_train_scrore = clf_tree.score(X_test, Y_test)
print("SCORE : ", clf_train_scrore)



## PLOTTING RESULTS

plot(Y_pred, "Classification tree", close = False)


## CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
# print('Confusion matrix\n\n', cm)

## SAVING PREDICTIONS

res = pd.DataFrame(Y_pred, index = get_indx(test_version), columns = ["tree"])
res.to_csv(f"{RES_PATH}/tree.csv", index = True)



