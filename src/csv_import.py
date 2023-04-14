## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"
exec(open(f'{tmp_path}/configs/config_local.py').read())

##
def import_raw_df(input_version:int=-1):
    if input_version == -1:
        df_name = f"{CSV_PATH}/{VERSION}_df.csv"
    else:
        df_name = f"{CSV_PATH}/{str(input_version)}_df.csv"
    # print("Reading :", df_name.split("/")[-1])
    try:
        df = pd.read_csv(df_name)
        df.index = df.pop("date")
        cols = df.columns.values
        df = df.dropna()
        X = df.loc[:, [col for col in cols if "crypto" not in col]]
        Y = df.loc[:, [col for col in cols if "crypto" in col]]
        del df

        # Between 0 and 1 (no negative)
        for c in [col for col in cols if 'score' in col] :
            X.loc[:, c] = (1 + X.loc[:, c]) / 2
        return X, Y
    except:
        raise Exception(f"Unable to import '{df_name}'.")

def import_several_csv(files:list, to_np:bool=True):
    X, Y = import_raw_df(files[0])
    for file in files[1:]:
        try :
            x, y = import_raw_df(file)
            X, Y = pd.concat([X, x]), pd.concat([Y, y])
        except:
            print(f"Couldn't upload file {file}")
    if to_np:
        return np.array(X), np.array(Y)
    return X, Y

##

def preprocess_df(X_, Y_, threshold:float=0.1, to_np:bool=True, norm:bool=True):
    #Copy obj
    X, Y = X_.copy(), Y_.copy()

    if isinstance(X, np.ndarray):
        tmp = X[1:,:]
        X = np.diff(X, axis=0)
        X[np.isneginf(X)] = -1
        X[np.isposinf(X)] = 1
        if norm :
            X = (X - X.min(axis=0)) / X.ptp(axis=0)
        if threshold > 0 :
            Y[Y>=threshold] = 1
            Y[Y< threshold] = 0
        Y = Y[1:, :] # to match X

    elif isinstance(X, pd.DataFrame):
        X = X.diff(axis=0).dropna()
        # X = X.pct_change().fillna(0)#.dropna()
        X[np.isneginf(X)] = -1
        X[np.isposinf(X)] = 1
        if norm :
            X = (X - X.min())/(X.max() - X.min())
        if threshold > 0 :
            Y.mask(Y>=threshold, 1, inplace=True)
            Y.mask(Y< threshold, 0, inplace=True)
        Y = Y.iloc[1:, :] # to match X

    tmp = np.array(Y)

    # print(f" > {len(tmp[tmp==1])}/{len(tmp)} ones after preprocessing.")

    if to_np:
        return np.array(X), np.array(Y)
    return X, Y

##

def get_indx(input_version:list=[-1]) :
    indx = []
    for f in input_version :
        try :
            df = pd.read_csv(f"{CSV_PATH}/{f}_df.csv")
            tmp = list(df["date"])
            try :
                if indx[-1] == tmp[0] :
                    tmp.pop(0)
            except :
                _=1
            indx += tmp
        except:
            raise Exception(f"Unable to import '{df_name}'.")
    return indx
