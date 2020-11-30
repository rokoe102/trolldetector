import pandas as pd
import numpy as np
from pathlib import Path
import os

# load datasets and return a list of all tweets (text only)
def prepare_datasets():
    root = Path(__file__).parent.parent
    troll_path = os.path.join(root, "datasets", "troll_top10.csv")
    nontroll_path = os.path.join(root, "datasets", "nontroll_top10.csv")

    troll = pd.read_csv(troll_path, encoding="utf-8", low_memory=False)
    nontroll = pd.read_csv(nontroll_path, encoding="utf-8", low_memory=False)

    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()
    tweets.extend(nontroll)

    return tweets

# return the correct labels for the elements in the dataset
def getTarget():
    target = np.full(303036, "troll").tolist()
    t = np.full(324873, "nontroll").tolist()
    target.extend(t)

    return target