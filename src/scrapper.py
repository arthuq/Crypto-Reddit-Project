
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

##


"""
This scrapper will exctract data from reddit, using specified subreddit names and
start/end dates from the search.
It will compute several quantities (number of posts, comments, NLP analysis score...)
and save everything into a csv file, in the "csv" folder.

"""


## VARIABES
'''
#START_TIME, END_TIME = "15/11/2022", "15/01/2023"    # JOUR MOIS ANNEE
START_TIME, END_TIME = "01/02/2023", "07/02/2023"    # JOUR MOIS ANNEE


SUBREDDIT_LIST = ["wallstreetbets", "Bitcoin", "CryptocurrencyMemes", "BitcoinMarkets"]
# ["CryptocurrencyMemes"]
# ["wallstreetbets", "Cryptocurrency", "CryptoMarkets", "Bitcoin", "BitcoinBeginners", "CryptocurrencyMemes","CryptoTechnology","BitcoinMarkets"]


CRYPTO_LIST = ["BTC", "ETH"]
# ["BTC", "ETH", "LTC", "XRP"]
'''

## VERSION OF DOWNLOAD

def update_version(start_time:str, end_time:str, subreddit_list:list):
    global version
    with open(version_path, 'r') as f:
        try :
            last_line = f.readlines()[-1]
        except:
            last_line = "0"
        version = str(int(last_line.split(",")[0]) + 1)
        new_line = version + "," + start_time.replace("-","/") + "," + end_time.replace("-", "/") + "," + "".join([s+'-' for s in subreddit_list])[:-1]

    with open(version_path, 'a') as f:
        try:
            f.write(new_line + "\n")
        except:
            raise Exception("Unable to write in _version.txt")


'''
version + "," + start_time.replace("-","/") + "," + end_time.replace("-", "/") + "," + "".join([s+'-' for s in subreddit_list])[:-1]
'''

def get_version():
    global version
    with open(version_path, 'r') as f:
        try :
            last_line = f.readlines()[-1]
        except:
            last_line = "0"
        version = str(int(last_line.split(",")[0]) + 1)


## CREATING EMPTY DF
def create_date_index(start_time:str, end_time:str, type:str):
    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y")
    data = pd.DataFrame()
    date_index = pd.date_range(start, end, freq='d').astype(str)
    data["date"] = date_index
    data.index = date_index
    data.index.name = "date"
    del data["date"]
    data["_name"] = type
    return data

## FUNCTIONS

def get_score(txt_file_name:str, subreddit:str):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    sia = SIA()

    #getting file and corpus words
    tmp = pd.read_csv(txt_file_name, sep=";", header=None, names=["date","content"], lineterminator='\n')
    pos_file = open(pos_corpus_path, "r")
    neg_file = open(neg_corpus_path, "r")
    pos_corpus = set(pos_file.read().split("\n"))
    neg_corpus = set(neg_file.read().split("\n"))

    def polarity(line):
        try :
            return sia.polarity_scores(line)["compound"]
        except:
            return 0

    def buysell(line):
        try :
            line = set( line.lower().split() )
            pos_count = 0.5*len(line & pos_corpus)/len(pos_corpus)
            neg_count = 0.5*len(line & neg_corpus)/len(pos_corpus)
            return 0.5 + pos_count - neg_count
        except:
            return 0.5

    if "comments" in txt_file_name:
        title = "score_comments_"
    elif "titles" in txt_file_name:
        title = "score_titles_"
    else:
        title = "score_"
    title += subreddit

    tmp[title] = tmp["content"].apply(polarity)
    tmp[title] = (tmp[title]+1)/2
    title = "corpus"+ title[5:]
    tmp[title] = tmp["content"].apply(buysell)
    del tmp["content"]
    tmp = tmp.groupby(['date']).mean()
    # print(f"Got some errors in polarity computation for {txt_file_name}.")
    return tmp


## FUNCTION GET COUNT
def get_count(txt_file_name:str, subreddit:str):
    tmp = pd.read_csv(txt_file_name, sep=";", header=None,  names=["date","content"],lineterminator='\n')
    tmp = tmp.groupby(['date']).size().reset_index()

    if "comments" in txt_file_name:
        title = "count_comments_"
    elif "titles" in txt_file_name:
        title = "count_titles_"
    else:
        title = "count_"
    title += subreddit
    tmp.columns = ["date", title]
    tmp.index = tmp["date"]
    del tmp["date"]
    return tmp

## FUNCTION DOWNLOADING DATA

def downloadFromUrl(filename:str, object_type:str, subreddit:str):
    # print(f"Saving {object_type}s to {filename}")
    count = 0
    if convert_to_ascii:
        handle = open(filename, 'w', encoding='ascii')
    else:
        handle = open(filename, 'w', encoding='UTF-8')
    previous_epoch = int(end.timestamp())
    break_out = False

    while True:
        new_url = url.format(object_type, filter_string) + str(previous_epoch)
        json_text = requests.get(new_url, headers={'User-Agent': "Post downloader by /u/Watchful1"})
        time.sleep(1)
        try:
            json_data = json_text.json()
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            continue

        if 'data' not in json_data:
            break

        objects = json_data['data']
        if len(objects) == 0:
            break

        for object in objects:
            previous_epoch = object['created_utc'] - 1
            if start is not None and datetime.utcfromtimestamp(previous_epoch) < start:
                break_out = True
                break
            count += 1

            if object_type == 'comment': #----------------------------------
                try:
                    handle.write(datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"))
                    handle.write(";" + object['body'].replace("\n"," "))
                    handle.write("\n")

                except Exception as err:
                    print(f"{'X '*30} Couldn't print comment: https://www.reddit.com{object['permalink']}")
                    print(traceback.format_exc())
            # --------------------------------------------------------------
            elif object_type == 'submission': #-----------------------------
                try:
                    handle.write(datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"))
                    handle.write(";" + object['title'].replace("\n"," "))
                    handle.write("\n")

                except Exception as err:
                    print(f"{'X '*30} Couldn't print post: {object['url']}")
                    print(traceback.format_exc())
            #---------------------------------------------------------------
        if break_out:
            break
    print(f"Saved {count} {object_type}s from {subreddit}")
    handle.close()


##
## GETTING COMMENTS AND COUNTS
##

def download(start_time:str, end_time:str, subreddit_list:list, crypto_list:list) :
    global start, end, convert_to_ascii, filter_string, url

    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y") + timedelta(days=1)

    data_crypto = create_date_index(start_time, end_time, "crypto")
    data_count = create_date_index(start_time, end_time, "count")
    data_scores = create_date_index(start_time, end_time, "score")

    #download cotnent for each subreddit
    for subreddit in subreddit_list :
        username = ""
        thread_id = ""
        convert_to_ascii = False

        filter_string = None
        if username == "" and subreddit == "" and thread_id == "":
            print("Fill in username, subreddit or thread id")
            sys.exit(0)

        filters = []
        if username:
            filters.append(f"author={username}")
        if subreddit:
            filters.append(f"subreddit={subreddit}")
        if thread_id:
            filters.append(f"link_id=t3_{thread_id}")
        filter_string = '&'.join(filters)

        url = "https://api.pushshift.io/reddit/{}/search?limit=1000&order=desc&{}&before="

        #File name
        fname = subreddit + "_" + start_time.replace("/","-") + "_" + end_time.replace("/","-")

        #Dowload function, temporary save comments and titles in .txt file
        if not thread_id:
            # downloadFromUrl(f"{version}_{fname}_titles.txt", "submission")
            downloadFromUrl(f"{csv_path}/titles.txt", "submission", subreddit)
            print("download titles done")

        # downloadFromUrl(f"{version}_{fname}_comments.txt", "comment")
        downloadFromUrl(f"{csv_path}/comments.txt", "comment", subreddit)
        print("download comments done")

        #Getting scores of comments and titles
        data_scores = data_scores.join( get_score(f"{csv_path}/comments.txt",subreddit) )
        data_scores = data_scores.join( get_score(f"{csv_path}/titles.txt",subreddit) )
        print("score done")

        #Getting count of comments and titles
        data_count = data_count.join(get_count(f"{csv_path}/comments.txt", subreddit))
        data_count = data_count.join(get_count(f"{csv_path}/titles.txt", subreddit))
        print("count done")

        #removing txt files with comments and titles
        os.remove(f'{csv_path}/comments.txt')
        os.remove(f'{csv_path}/titles.txt')

    # GETTING CRYPTOS DATA
    start_time_tmp, end_time_tmp = start_time.replace("/","-"), end_time.replace("/","-")
    for crypto in crypto_list :
        try :
            scraper = CmcScraper(crypto, start_time_tmp, end_time_tmp)
            tmp = scraper.get_dataframe()
            tmp = tmp[['Date', 'Close']]
            tmp.columns = ["date", f"{crypto}"]
            tmp.index = tmp["date"]
            del tmp["date"]
            tmp = tmp.iloc[::-1]
            data_crypto = data_crypto.join(tmp)

        except:
            print(f"Unable to scrape {crypto} data.")
            continue
            # raise Exception("Unable to scrape crypto data.")

    #Changing indexes
    data_scores.index.name = "date"
    data_count.index.name = "date"
    data_crypto.index.name = "date"

    try :
        df = pd.concat([data_crypto, data_scores, data_count], axis=1)
        df.to_csv(f"{csv_path}/{version}_df.csv", index=True)
        print(f"Saved file as {version}_df.csv")
    except:
        raise Exception("Unable to concatenate the 3 dataframes.")

# **********************************************************************
# **********************************************************************
def main_api(start_time:str, end_time:str, subreddit_list:list, crypto_list:list) :
    import os
    project_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"
    exec(open(f'{project_path}/configs/config_local.py').read())

    #version update of the files
    get_version()
    print(f"{'*'*12} version : {version} {'*'*12}")

    #lauching the download
    download(start_time, end_time, subreddit_list, crypto_list)
    update_version(start_time, end_time, subreddit_list)

# **********************************************************************
# **********************************************************************

## LAUCH MAIN

# START_TIME, END_TIME = "02/02/2023", "05/02/2023"
# CRYPTO_LIST = ["BTC"]
# SUBREDDIT_LIST = ["BitcoinMarkets"]

# main_api(START_TIME, END_TIME, SUBREDDIT_LIST, CRYPTO_LIST)


##
# ["BTC", "ETH", "LTC", "XRP"]
# ["wallstreetbets", "Cryptocurrency", "CryptoMarkets", "Bitcoin", "BitcoinBeginners", "CryptocurrencyMemes","CryptoTechnology","BitcoinMarkets"]

