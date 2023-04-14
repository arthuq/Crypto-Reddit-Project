

"""
import requests
from datetime import datetime
import traceback
import time
import json
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from cryptocmd import CmcScraper
"""


## CONFIGURATION

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

# Execute scrapper file
exec(open(f'{SRC_PATH}/scrapper.py').read())


#exec(open(f'{SRC_PATH}/analyze.py').read())

# Importing py files
#os.chdir(SRC_PATH)
#from scrapper import *
#from analyze import *


### GETTING CRYPTOS DATA
"""
def get_key_dates(start_time:str, end_time:str, crypto_list:list, threshold:float=0.3):
    tmp_crypto = create_date_index(start_time, end_time, "c")
    del tmp_crypto["_name"]
    start_time, end_time = start_time.replace("/", "-"), end_time.replace("/", "-")

    for crypto in crypto_list :
        try :
            scraper = CmcScraper(crypto, start_time, end_time)
            tmp = scraper.get_dataframe()[['Date', 'Close']]
            #tmp = tmp[['Date', 'Close']]
            tmp.columns = ["date", f"{crypto}"]
            tmp.index = tmp["date"]
            del tmp["date"]
            tmp = tmp.iloc[::-1]
            tmp_crypto = tmp_crypto.join(tmp)
        except:
            print(f"Unable to scrape '{crypto}' data.")

    tmp_crypto.index.name = "date"
    tmp_crypto = tmp_crypto.pct_change().fillna(0)
    times = get_key_times(tmp_crypto, threshold, False)

    out, tmp, to_del = pd.DataFrame(times), list(times), []
    for i, (one, two) in enumerate(zip(tmp[:-1], tmp[1:])) :
        if ((two-one).days < 7) :
            to_del.append(i)
    out.drop(to_del, axis=0, inplace=True)
    out = out.reset_index(drop=True)
    out["start"] = out["date"] - timedelta(days=7)

    start = [x.strftime("%d/%m/%Y") for x in list(out["start"])]
    end = [x.strftime("%d/%m/%Y") for x in list(out["date"])]
    return start, end
"""
##

def iterative_dates(start_time, end_time):
    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y")
    t1 = pd.date_range(start, end, freq='m')
    t0 = t1 + timedelta(days = 1)
    t0, t1 = list(t0.astype(str))[:-1], list(t1.astype(str))
    t0 = [start_time]+["/".join(s.split("-")[::-1]) for s in t0]
    t1 = ["/".join(s.split("-")[::-1]) for s in t1]
    return t0, t1

##

def main_build(start_time:str, end_time:str, crypto_list:list, subreddit_list:list, threshold:float=0.03):

    tmp = 'n' #input("Select only jump periods in crypto ? (y/n) : ")
    if tmp == "y" :
        return "Can't, disabled."
        start, end = get_key_dates(start_time, end_time, crypto_list, threshold)
        print("Found dates for crypto jumps :")
        print(end)
        a = input("Continue ? (y/n) : ")
        if a != "y":
            return 0
        for t0, t1 in zip(start, end):
            print(f"{'*'*55}\n{t0} to {t1}\n{'*'*55}")
            try :
                main_api(t0, t1, subreddit_list, crypto_list)
                main_analyze()
            except:
                print("error running api")

    else:
        h = input("Use segmented time dates ? (y/n) : ")
        if h == "y":
            t0_, t1_ = iterative_dates(start_time, end_time)
            for t0, t1 in zip(t0_, t1_):
                main_api(t0, t1, subreddit_list, crypto_list)
                print("_"*35)
        else:
            main_api(start_time, end_time, subreddit_list, crypto_list)
    print("Finished all api requests.")


## LAUCH MAIN FUNCTION

START_TIME, END_TIME = "01/02/2023", "01/04/2023"
CRYPTO_LIST = ["BTC"]
SUBREDDIT_LIST = ["CryptocurrencyMemes", "wallstreetbets", "Cryptocurrency"]

main_build(START_TIME, END_TIME, CRYPTO_LIST, SUBREDDIT_LIST)


# ", "ETH", "LTC", "XRP"]
# ["wallstreetbets", "Cryptocurrency", "CryptoMarkets", "Bitcoin", "BitcoinBeginners", "CryptocurrencyMemes","CryptoTechnology","BitcoinMarkets"]

##




