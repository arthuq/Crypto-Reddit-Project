
#Importing libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing files
# os.chdir('D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT/API')
# from API import *


##SCRAPPING NEW FILES WITH API


START_TIME, END_TIME = "01/02/2023", "07/02/2023"

CRYPTO_LIST = ["BTC", "ETH", "LTC", "XRP"]
# ["BTC", "ETH", "LTC", "XRP"]

SUBREDDIT_LIST = ["Bitcoin"]
# ["wallstreetbets", "Cryptocurrency", "CryptoMarkets", "Bitcoin", "BitcoinBeginners", "CryptocurrencyMemes","CryptoTechnology","BitcoinMarkets"]

# main(START_TIME, END_TIME, SUBREDDIT_LIST, CRYPTO_LIST)


## GETTING LAST VERSION
'''
def get_version():
    with open('D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT/API/DATA/0_version.txt', 'r') as f:
        last_line = f.readlines()[-1]
    return last_line.split(",")[0]
'''

def import_last_df_raw(input_version=-1):
    df_name = csv_path
    if input_version == -1:
        df_name += "/" + version + "_df.csv"
    else:
        df_name += "/" + str(input_version) + "_df.csv"
    try:
        df = pd.read_csv(df_name)
    except:
        raise Exception(f"file '{df_name}' does not exist.")
    df.index = df["date"]
    del df["date"]
    return df

## SEPARATING DATA

def sep_df(df):
    ind = [i  for i,c in enumerate(df.columns.values) if "_name" in c]
    crypto = df.iloc[:,1:ind[1]]
    scores = df.iloc[:,ind[1]+1:ind[2]]
    counts = df.iloc[:,ind[2]+1:]
    # del crypto["name"], scores["name.1"], counts["name.2"]
    return crypto, scores, counts

def import_df(version):
    df = import_last_df_raw(version)
    cp, sc, ct = sep_df(df)
    all = sc.join(ct)
    return {"crypto":cp, "score":sc, "count":ct, "all":all}

## PERCENT CHANGE

def pct_change(df):
    if type(df)==dict:
        for k in df.keys():
            df[k] = df[k].pct_change().fillna(0)
    elif type(df)==pandas.core.frame.DataFrame :
        df = df.pct_change().fillna(0)
    return df

## KEY DATES OF BIG CRYPTO CHANGES IN PRICE

def get_key_times(crypto, threshold, sep=False):
    n = len(crypto.columns)
    """
    sep : specifies if we want to seperate up and down jumps.
    """
    if sep==True:
        key_times_up = crypto[crypto > threshold]
        key_times_down = crypto[crypto < -threshold]

        key_times_up = key_times_up.groupby(['date']).count().sum(axis=1)
        key_times_up = key_times_up[key_times_up >= max(1, n//2) ].index

        key_times_down = key_times_down.groupby(['date']).count().sum(axis=1)
        key_times_down = key_times_down[key_times_down >= max(1, n//2)].index
        return key_times_up, key_times_down #not good, dimenson of return changes.

    else :
        key_times = crypto[abs(crypto) > threshold]
        key_times = key_times.groupby(['date']).count().sum(axis=1)
        key_times = key_times[key_times >= max(1, n//2) ].index
        return key_times


## COMPUTING COR

def get_correlation(df):
    cor = pd.DataFrame()
    for crypto in df["crypto"]:
        cor[crypto] = df["all"].corrwith(df["crypto"][crypto])
    return cor

## PLOTTING
def save_summary(df, cor, key_times_up, key_times_down, version_input):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    fig.tight_layout()
    ax1, ax2, ax3, ax4 = ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]

    try:
        ax1.plot(df["crypto"], "o-")
        ax1.legend(df["crypto"].columns, loc='upper left')
        ax1.set_title("Crypto pct change")
        tmp = round( 1.1*np.max(abs(df["crypto"]).values), 5)
        ax1.set_ylim( -tmp,tmp)
        ax1.grid()
    except:
        print("Couldn't plot crypto data.")

    try:
        ax2.plot(df["count"], "o-")
        ax2.legend(df["count"].columns, loc='upper left')
        ax2.set_title("Count pct change")
        ax2.set_ylim(-1,1)
        ax2.grid()
    except:
        print("Couldn't plot count data.")

    try:
        ax3.plot(df["score"], "o-")
        ax3.legend(df["score"].columns, loc='upper left')
        ax3.set_title("Score pct change")
        tmp = round( 1.1*np.max(abs(df["score"]).values), 5)
        ax3.set_ylim( -tmp,tmp)
        ax3.grid()
    except:
        print("Couldn't plot score data.")

    for ax in (ax1, ax2, ax3):
        for kt in key_times_up :
            ax.axvline(x = kt , linewidth = 2, color='green' )
        for kt in key_times_down :
            ax.axvline(x = kt , linewidth = 2, color='red' )

    try :
        cor.plot(kind="bar", ax=ax4)
        ax4.grid()
        ax4.set_ylim(-1,1)
        ax4.set_title("correlations")
    except:
        print("Couldn't plot correlation data.")

    plt.gcf().autofmt_xdate()
    if version_input == -1:
        version_input = version
    ax4.text(-0.9, 3.3, f"Version : {version_input}", fontsize=16, verticalalignment='center', horizontalalignment = 'center')
    # plt.show()
    plt.savefig(f'{fig_path}/{str(version_input)}_summary.png')
    print(f"file '{str(version_input)}_summary.png' saved.")

##

def main_analyze(version=-1):
    import os
    project_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"
    exec(open(f'{project_path}/configs/config_local.py').read())

    # os.chdir(src_path)
    # from scrapper import *
    # get_version()

    df = import_df(version)
    df = pct_change(df)
    key_times_up, key_times_down = get_key_times(df["crypto"], 0.01, True)
    cor = get_correlation(df)
    save_summary(df, cor, key_times_up, key_times_down, version)

##
# RUN:
main_analyze()