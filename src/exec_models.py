## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Importing py files
# exec(open(f'{SRC_PATH}/scrapper.py').read())
exec(open(f'{SRC_PATH}/csv_import.py').read())

##

## IMPORTING THE TRAINING AND TEST DATA

# parameters
global train_version, test_version, jump_threshold, NORM

train_version = list(range(24, 35))
test_version = [35, 36]
jump_threshold = 0.06
NORM = True

# model results
_, Y_test = import_several_csv(test_version, to_np = False)

Y_test_ = Y_test.copy().drop_duplicates()

top = (np.mean(Y_test_) + np.std(Y_test_))[0]
bot = (np.mean(Y_test_) - np.std(Y_test_))[0]

top, bot = float(top), float(bot)

_, Y_test = preprocess_df(_, Y_test, jump_threshold, to_np = True, norm = NORM)

## EXECUTION DES MODELS

model_files = [f for f in os.listdir(SRC_PATH) if "model_" in f]
print(f"There are {len(model_files) + 2} models")
# +2 because 2 modre models in one file

## EXECUTING ALL MODELS
for mod in model_files :
    print(mod, "_"*50)
    exec(open(f'{SRC_PATH}/{mod}').read())

## CONSTRUCTION DF
models = pd.DataFrame(0, index = get_indx(test_version), columns = os.listdir(RES_PATH))
for file in os.listdir(RES_PATH) :
    tmp = pd.read_csv(f"{RES_PATH}/{file}")
    tmp = tmp.iloc[:,1]
    models.loc[:,file] = np.array(tmp)

##
def discretize2(x:int):
    if x >= float(top) :
        return 1.0
    elif x <= float(bot) :
        return 0.0
    else :
        return 0.5

def discretize(x:int):
    if x >= 0.75 : return 1.0
    elif x <= 0.25 : return 0.0
    else : return 0.5

def discretize_two(x:int):
    if x >= 0.65 :
        return 1.0
    else:
        return 0.0

## RESULTS

# discretization of all models
disc_models = models.applymap(discretize_two)

# into arrays
yy, dc = np.array(Y_test), np.array(disc_models)

# Counting accuracy of good label
cc = (yy==dc).astype(int)
tmp = np.sum(cc, 0) / len(cc)
acc = {mod:round(sc, 3) for mod,sc in zip(models.columns, tmp)}
del tmp



## CATEGORISATION PAR MOYENNE

means = models.mean(axis=1)

means_d = means.map(discretize)

# means_nm = means * 0.5 / means.mean() # Nouvelle moyenne à 0.5

means_nm = (means_nm - means_nm.min()) / (means_nm.max() - means_nm.min())


## CATEGORIZATION PAR MAJORITE

def cat_maj(line):
    out = {0:0, 0.5:0, 1:0}
    for i, j in line.value_counts().items() :
        out[i] = j
    return max(out, key=out.get)

majority = disc_models.apply(cat_maj, 1)

##



def get_score(df) :


    # into arrays
    yy, dc = np.array(Y_test), np.array(df)

    # Counting accuracy of good label
    cc = (yy==dc).astype(int)
    print(cc)

    tmp = np.sum(cc, 0) / len(cc)
    print(tmp)
    return tmp


get_score(majority)

## PLOT

# models.plot(style = "x", alpha=0.3, title="Agregation of models")

accuracy_maj = 0
accuracy_moy = 0

m1, m2 = np.array(majority), np.array(means_nm)

plt.plot(m1, "^", color="blue", label = f"Majority")
plt.plot(m2, "*", color="red", label = f"Mean")

# DISCRETIZATION OR REAL OUTPUT Y_TEST

y = np.array(Y_test_).flatten()
f = np.vectorize(discretize2)
y = f(y)

y = yy
for i in list(np.where(y == 1)[0]) :
    plt.axvspan(i-.5, i+.5, 0.75, 1, facecolor='limegreen', alpha=0.5)

for i in list(np.where(y == 0.5)[0]) :
    plt.axvspan(i-.5, i+.5,0.25, 0.75, facecolor='gold', alpha=0.5)

for i in list(np.where(y == 0)[0]) :
    plt.axvspan(i-.5, i+.5,0, 0.25, facecolor='salmon', alpha=0.5)

plt.title("Agragation of models")

plt.ylim([-0.1,1.1])
plt.legend()
plt.show()
plt.grid()
plt.savefig(f'{tmp_path}/photos/agregate.png')

##


