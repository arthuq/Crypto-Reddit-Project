"""
Arthur MARON
2023
"""
"""
import sys
import requests
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

#PATHS
global project_path, config_path, src_path, dat_path, csv_path,fig_path, version_path, neg_corpus_path, pos_corpus_path, version

project_path = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

config_path = project_path + "/configs"

src_path = project_path + "/src"

dat_path = project_path + "/src/dat"

csv_path = src_path + "/csv"
fig_path = src_path + "/fig"

version_path = dat_path + "/_version.txt"
neg_corpus_path = dat_path + "/_neg_corpus.txt"
pos_corpus_path = dat_path + "/_pos_corpus.txt"

with open(version_path, 'r') as f:
    try :
        last_line = f.readlines()[-1]
    except:
        last_line = "0"
    version = str(int(last_line.split(",")[0]))