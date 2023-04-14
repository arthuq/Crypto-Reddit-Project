
## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# # Importing py files
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
X_train, Y_train = preprocess_df(X_train, Y_train, jump_threshold, to_np= True, norm = NORM)
X_test, Y_test = preprocess_df(X_test, Y_test, jump_threshold, to_np= True, norm = NORM)

## FINDING BEST MODEL FOREST


param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}

rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

rand_search.fit(X_train, Y_train.flatten())

# best mod
best_rf = rand_search.best_estimator_

# bets parameters
print('Best hyperparameters:',  rand_search.best_params_)

## FITTING BEST ESTIMATOR

# fitting
best_rf.fit(X_train, Y_train.flatten())
Y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy Forest (test set) :", accuracy)

## PLOT

plt.plot(Y_pred, "o", label = "Random Forest Classification", markersize=8)

top = np.mean(Y_test_) + np.std(Y_test_)
bot = np.mean(Y_test_) - np.std(Y_test_)
for i in list(np.where(Y_test_ > top)[0]) :
    plt.axvspan(i-.5, i+.5, 0.75, 1, facecolor='limegreen', alpha=0.5)

for i in set(list(np.where(Y_test_ > bot))[0]) & set(list(np.where(Y_test_ < top))[0]) :
    plt.axvspan(i-.5, i+.5,0.25, 0.75, facecolor='gold', alpha=0.5)

for i in list(np.where(Y_test_ < bot)[0]) :
    plt.axvspan(i-.5, i+.5, 0, 0.25, facecolor='salmon', alpha=0.5)

plt.title(f"Random Forest Classification\nVersion {train_version}")
# plt.xticks(rotation=45)
plt.grid()
plt.legend()
# plt.show()

plt.savefig(f'{tmp_path}/photos/forest.png')
plt.close()

pd.DataFrame(Y_pred).to_csv(f"{RES_PATH}/Random forest.csv", index = True)