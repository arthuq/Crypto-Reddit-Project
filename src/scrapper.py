## DESCRTIPTION

"""
This scrapper will exctract data from reddit, using specified subreddit names and
start/end dates from the search.
It will compute several quantities (number of posts, comments, NLP analysis score...)
and save everything into a csv file, in the "csv" folder.
"""
## IMPORTATION OF CONFIG FILE

# Selecting dir path for config file
import os
tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

# Execute config file
exec(open(f'{tmp_path}/configs/config_local.py').read())

## VERSION OF DOWNLOAD

def update_version(start_time:str, end_time:str, crypto_list:list, subreddit_list:list):
    """Writes a new line in version file corresponding to the new downloaded csv"""
    # Reading the version
    with open(VERSION_PATH, 'r') as f:
        try :
            last_line = f.readlines()[-1]
            up_version = str(int(last_line.split(",")[0]) + 1)
        except:
            up_version = "1"

    times = start_time.replace("-","/") + "," + end_time.replace("-", "/")
    cryptos = "-".join(crypto_list)
    subreddits = "-".join(subreddit_list)
    new_line = ",".join([up_version, times, cryptos, subreddits])

    # Writing new version
    with open(VERSION_PATH, 'a') as f:
        try:
            f.write(new_line + "\n")
        except:
            print(new_line)
            raise Exception("Unable to write in _version.txt")

    # Re-executing config files to have new global variables up to date.
    try:
        exec(open(f'{CONFIG_PATH}/config_local.py').read())
    except:
        print("Unable to execute config file after updated of version.txt.")


def get_version():
    """Checking last version in version file"""
    with open(VERSION_PATH, 'r') as f:
        try :
            last_line = f.readlines()[-1]
            return str(int(last_line.split(",")[0]) + 1)
        except:
            return "1"


## CREATING EMPTY DF
""" old
def create_date_index(start_time:str, end_time:str, type:str):
    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y")
    indx = pd.date_range(start, end, freq='d').astype(str)
    out = pd.DataFrame(indx)
    out.index, out.index.name, out["_name"] = indx, "date", type
    out.drop(columns=out.columns[0], axis=1,  inplace=True)
    return out
"""

def create_date_index(start_time:str, end_time:str):
    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y")
    indx = pd.date_range(start, end, freq='d').astype(str)
    return pd.DataFrame(index=indx)


## FUNCTIONS

def get_score(txt_file_name:str):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    sia = SIA()
    tmp = pd.read_csv(txt_file_name, sep=";", header=None, names=["date","content"], lineterminator='\n', quoting=3, error_bad_lines=False)

    pos_file = open(POS_CORPUS_PATH, "r")
    pos_corpus = set(pos_file.read().split("\n"))
    neg_file = open(NEG_CORPUS_PATH, "r")
    neg_corpus = set(neg_file.read().split("\n"))

    def polarity_analysis(line):
        try :
            return sia.polarity_scores(line)["compound"]
        except:
            return 0.0

    def corpus_analysis(line):
        try :
            line = set( line.lower().split() )
            pos_count = len(line & pos_corpus)/len(pos_corpus)
            neg_count = len(line & neg_corpus)/len(pos_corpus)
            return pos_count-neg_count
        except:
            return 0.0

    title = txt_file_name.split("/")[-1].split(".")[0] + "_score"
    corp = title.replace("titles", "corpus")

    tmp[title] = tmp["content"].apply(polarity_analysis)
    tmp[corp] = tmp["content"].apply(corpus_analysis)
    del tmp["content"]
    tmp = tmp.groupby(['date']).mean()
    #tmp[title] = (tmp[title]+1)/2 # renormalisation entre 0 et 1, au lieu de -1,1

    return tmp.round(decimals = 5)


## FUNCTION GET COUNT
def get_count(txt_file_name:str):
    tmp = pd.read_csv(txt_file_name, sep=";", header=None, names=["date","content"],lineterminator='\n', quoting=3, error_bad_lines=False)
    tmp = tmp.groupby(['date']).size().reset_index()
    title = txt_file_name.split("/")[-1].split(".")[0] + "_count"
    tmp.columns = ["date", title]
    tmp.index = tmp["date"]
    del tmp["date"]
    #tmp = tmp/tmp.sum() # Normalisation
    # tmp = tmp.round(decimals = 5)
    return tmp

## FUNCTION DOWNLOADING DATA

def downloadFromUrl(filename:str, object_type:str, subreddit:str):
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
    print(f" > Scrapped  {count} {object_type}s from {subreddit}")
    handle.close()


##
## GETTING COMMENTS AND COUNTS
##

def download(start_time:str, end_time:str, subreddit_list:list, crypto_list:list) :
    global start, end, convert_to_ascii, filter_string, url

    start = datetime.strptime(start_time, "%d/%m/%Y")
    end = datetime.strptime(end_time, "%d/%m/%Y") + timedelta(days=1)

    data_crypto = create_date_index(start_time, end_time)
    data_count = create_date_index(start_time, end_time)
    data_scores = create_date_index(start_time, end_time)

    # SCRAPE FOR EACH SUBREDDIT =================================================
    for subreddit in subreddit_list :
        username, thread_id, convert_to_ascii, filter_string = "", "", False, None
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

        #Dowload function, temporary save comments and titles in .txt file
        if not thread_id:
            downloadFromUrl(f"{TXT_PATH}/{subreddit}_titles.txt", "submission", subreddit)
        downloadFromUrl(f"{TXT_PATH}/{subreddit}_comments.txt", "comment", subreddit)

    # GETTING REDDIT COUNT AND SCORES ===========================================
    for subreddit in subreddit_list :
        #Getting scores of comments and titles
        data_scores = data_scores.join( get_score(f"{TXT_PATH}/{subreddit}_comments.txt") )
        data_scores = data_scores.join( get_score(f"{TXT_PATH}/{subreddit}_titles.txt") )
        print(f" > Extracted {subreddit} scores.")

        #Getting count of comments and titles
        data_count = data_count.join(get_count(f"{TXT_PATH}/{subreddit}_comments.txt"))
        data_count = data_count.join(get_count(f"{TXT_PATH}/{subreddit}_titles.txt"))
        print(f" > Extracted {subreddit} counts.\n")

    # GETTING CRYPTOS DATA =====================================================
    start_time_tmp, end_time_tmp = start_time.replace("/","-"), end_time.replace("/","-")
    for crypto in crypto_list :
        scraper = CmcScraper(crypto, start_time_tmp, end_time_tmp)
        tmp = scraper.get_dataframe().iloc[::-1]
        var = (tmp["High"]-tmp["Low"]) / abs((tmp["Close"] + tmp["Open"])/2)
        var = np.round(var, 5)
        indx = list(tmp["Date"])
        del tmp
        c_name = crypto + "_crypto"
        data_crypto = data_crypto.join(pd.DataFrame(var.values, columns=[c_name], index = indx))

    # MERGING THE DATAFRAMES =====================================================
    data_scores.index.name = "date"
    data_count.index.name = "date"
    data_crypto.index.name = "date"
    try :
        df = pd.concat([data_crypto, data_scores, data_count], axis=1)
        df.to_csv(f"{CSV_PATH}/{NEXT_VERSION}_df.csv", index=True)
        print(f"Saved file as {NEXT_VERSION}_df.csv")
        for subreddit in subreddit_list :
            os.remove(f'{TXT_PATH}/{subreddit}_comments.txt')
            os.remove(f'{TXT_PATH}/{subreddit}_titles.txt')
        print("Removed txt files successfully.")
    except:
        raise Exception("Unable to concatenate the 3 dataframes.")


# **********************************************************************
def main_api(start_time:str, end_time:str, subreddit_list:list, crypto_list:list) :
    import os
    tmp_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"
    exec(open(f'{tmp_path}/configs/config_local.py').read())

    #lauching the download
    print(f"Lauching reddit api scrapper ({start_time} to {end_time}).")
    print("This will be version", NEXT_VERSION, "of the csv.\n")
    download(start_time, end_time, subreddit_list, crypto_list)
    update_version(start_time, end_time, crypto_list, subreddit_list)

# **********************************************************************

