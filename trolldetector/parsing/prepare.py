import pandas as pd
import numpy as np
from pathlib import Path
import os

# load datasets and return a list of all tweets (text only)
def prepare_datasets():
    root = Path(__file__).parent.parent
    troll_path = os.path.join(root, "datasets", "troll_tweets.csv")
    nontroll_path = os.path.join(root, "datasets", "nontroll_tweets.csv")


    troll = pd.read_csv(troll_path, engine="python", error_bad_lines=False, na_filter=False)
    nontroll = pd.read_csv(nontroll_path, engine="python", error_bad_lines=False, na_filter=False)

    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()

    tweets.extend(nontroll)

    return tweets, len(troll), len(nontroll)

# return the correct labels for the elements in the dataset
def getTarget(troll_len, nontroll_len):
    target = np.full(troll_len, "troll").tolist()
    t = np.full(nontroll_len, "nontroll").tolist()
    target.extend(t)

    return target