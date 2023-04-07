## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"
exec(open(f'{tmp_path}/configs/config_local.py').read())

##
def import_df_NN(input_version:int=-1, threshold:float=0.1):
    # Finding right csv name to download
    if input_version == -1:
        df_name = f"{CSV_PATH}/{VERSION}_df.csv"
    else:
        df_name = f"{CSV_PATH}/{str(input_version)}_df.csv"
    print("Reading :", df_name.split("/")[-1])

    try:
        # Importing df
        df = pd.read_csv(df_name)
        df.index = df["date"]
        del df["date"]
        cols = df.columns.values
        df = df.dropna()

        # Separating X and Y for y=f(X)
        indx = [1 if "crypto" in c else 0 for c in cols]
        indx_X, indx_Y = [i for i,j in enumerate(indx) if j==0], [i for i,j in enumerate(indx) if j==1]
        X, Y = df.iloc[:, indx_X], df.iloc[:, indx_Y]
        del df

        # Final processing for model
        X = X.pct_change().fillna(0) #.iloc[1:,]
        X[np.isneginf(X)] = -1
        X[np.isposinf(X)] = 1

        Y.mask(Y>=threshold, 1, inplace=True) # logistic regression output
        Y.mask(Y< threshold, 0, inplace=True)
        return np.array(X), np.array(Y), cols

    except:
        raise Exception(f"Unable to import '{df_name}'.")


def import_several_csv(files:list, threshold:float=0.5):
    if len(files) == 1:
        return import_df_NN(files[0], threshold)
    X, Y, cols = import_df_NN(files[0], threshold)

    for file in files[1:] :
        x, y, _ = import_df_NN(file, threshold)
        try :
            X = np.concatenate([X, x])
            Y = np.concatenate([Y, y])
        except:
            print(f"Couldn't upload file {file}")

    # Normalizing all count data
    """
    for i, c in enumerate([b for b in cols if "crypto" not in b]):
        if "count" in c :
            X[:,i] /= X[:,i].sum()
    """
    return X, Y
