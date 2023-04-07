"""
Arthur MARON
2023
"""

import os
import sys
import requests
import traceback
import time
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from cryptocmd import CmcScraper

from scipy import stats
from numpy.random import default_rng
rng = default_rng()
import torch
import torch.nn as nn

##


global PROJECT_PATH, CONFIG_PATH, SRC_PATH, DAT_PATH, CSV_PATH, FIG_PATH, TXT_PATH, VERSION_PATH, NEG_CORPUS_PATH, POS_CORPUS_PATH, VERSION, NEXT_VERSION

PROJECT_PATH = "D:/3.Cours EK/8. SEMESTRE DEUX/4. CRYPTO/PROJECT"

CONFIG_PATH = PROJECT_PATH + "/configs"
SRC_PATH = PROJECT_PATH + "/src"
DAT_PATH = PROJECT_PATH + "/src/dat"
CSV_PATH = SRC_PATH + "/csv"
FIG_PATH = SRC_PATH + "/fig"
TXT_PATH = SRC_PATH + "/txt_tmp"

VERSION_PATH = DAT_PATH + "/_version.txt"
NEG_CORPUS_PATH = DAT_PATH + "/_neg_corpus.txt"
POS_CORPUS_PATH = DAT_PATH + "/_pos_corpus.txt"

with open(VERSION_PATH, 'r') as f:
    try :
        last_line = f.readlines()[-1]
    except:
        last_line = "0"
    VERSION = str(int(last_line.split(",")[0]))
    NEXT_VERSION = str(1 + int(last_line.split(",")[0]))