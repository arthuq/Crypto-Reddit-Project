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

os.chdir('D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT/API')
from API import *
from PROCESS import *


### GETTING CRYPTOS DATA

def get_key_dates(start_time:str, end_time:str, crypto_list:list, threshold:float=0.3):
    tmp_crypto = create_date_index(start_time, end_time, "c")
    del tmp_crypto["name"]
    start_time, end_time = start_time.replace("/", "-"), end_time.replace("/", "-")

    for crypto in crypto_list :
        scraper = CmcScraper(crypto, start_time, end_time)
        tmp = scraper.get_dataframe()
        tmp = tmp[['Date', 'Close']]
        tmp.columns = ["date", f"{crypto}"]
        tmp.index = tmp["date"]
        del tmp["date"]
        tmp = tmp.iloc[::-1]
        tmp_crypto = tmp_crypto.join(tmp)
    del tmp
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

##

def main_build(start_time:str, end_time:str, crypto_list:list, subreddit_list:list, threshold:float=0.03):
    start, end = get_key_dates(start_time, end_time, crypto_list, threshold)
    print(start)
    a = input("Continue ? (y/n) : ")
    if a != "y":
        return 0
    for s, e in zip(start, end):
        print(f"----------------{s} to {e}----------------")
        try :
            main_api(s, e, subreddit_list, crypto_list)
            main_analyze()
        except:
            print("error running api")
    print("Finished all api requests.")

## LAUCH MAIN FUNCTION
START_TIME, END_TIME = "01/01/2023", "05/02/2023"

CRYPTO_LIST = ["BTC", "ETH"]
# ", "ETH", "LTC", "XRP"]
SUBREDDIT_LIST = ["Bitcoin", "CryptocurrencyMemes"]

# ["wallstreetbets", "Cryptocurrency", "CrypntoMarkets", "Bitcoin", "BitcoinBeginners", "CryptocurrencyMemes","CryptoTechnology","BitcoinMarkets"]

main_build(START_TIME, END_TIME, CRYPTO_LIST, SUBREDDIT_LIST, 0.03)
