import pandas as pd
import numpy as np

def prepare_datasets():
    troll = pd.read_csv("datasets/troll_top10.csv", encoding="utf-8", low_memory=False)
    nontroll = pd.read_csv("datasets/nontroll_top10.csv", encoding="utf-8", low_memory=False)

    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()
    tweets.extend(nontroll)

    return tweets

def getTarget():
    target = np.full(303036, "troll").tolist()
    t = np.full(324873, "nontroll").tolist()
    target.extend(t)

    return target