import pandas as pd
from sklearn import naive_bayes, neighbors, svm, tree, neural_network, preprocessing
from pathlib import Path
import os


# get a dict of hyperparameters from save file in order to use them
# for the comparison of all techniques
def load(technique):
    root = Path(__file__).parent.parent

    # instantiate classifier class and suitable parameters for technique

    params = []
    if technique == "KNN":
        params = pd.read_csv(os.path.join(root, "memory", "best_knn.csv"), index_col=0, header=None).T.to_dict("list")
        params = convert(params)
        clf = getattr(neighbors, params["clf"][0].replace("(", "").replace(")", ""))
        params["clf"] = [clf()]

    elif technique == "NB":
        params = pd.read_csv(os.path.join(root, "memory", "best_nb.csv"), index_col=0, header=None).T.to_dict("list")
        params = convert(params)
        clf = getattr(naive_bayes, params["clf"][0].replace("(", "").replace(")", ""))
        params["clf"] = [clf()]
        if params["scaling"][0] == "MinMaxScaler()":
            sca = getattr(preprocessing, params["scaling"][0].replace("(", "").replace(")", ""))
            params["scaling"] = [sca()]

    elif technique == "SVM":
        params = pd.read_csv(os.path.join(root, "memory", "best_svm.csv"), index_col=0, header=None).T.to_dict("list")
        params = convert(params)
        clf = getattr(svm, params["clf"][0].replace("(", "").replace(")", ""))
        params["clf"] = [clf()]

    elif technique == "tree":
        params = pd.read_csv(os.path.join(root, "memory", "best_tree.csv"), index_col=0, header=None).T.to_dict("list")
        params = convert(params)
        clf = getattr(tree, params["clf"][0].replace("(", "").replace(")", ""))
        params["clf"] = [clf()]

    elif technique == "MLP":
        params = pd.read_csv(os.path.join(root, "memory", "best_mlp.csv"), index_col=0, header=None).T.to_dict("list")
        params = convert(params)
        clf = getattr(neural_network, params["clf"][0].replace("(", "").replace(")", ""))
        params["clf"] = [clf()]
    else:
        print("Error: No valid classification technique")

    return params


# save best results of hyperparameter optimization as csv
def save(results, technique):
    root = Path(__file__).parent.parent

    df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(), results["params"])),
                      columns=["score", "params"])
    df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)

    # find maximum accuracy
    max_row = df[df["score"] == max(df["score"])]

    if technique == "KNN":
        max_row.T.to_csv(os.path.join(root, "memory", "best_knn.csv"), header=None, na_rep="None")
    elif technique == "NB":
        max_row.T.to_csv(os.path.join(root, "memory", "best_nb.csv"), header=None, na_rep="None")
    elif technique == "SVM":
        max_row.T.to_csv(os.path.join(root, "memory", "best_svm.csv"), header=None, na_rep="None")
    elif technique == "tree":
        max_row.T.to_csv(os.path.join(root, "memory", "best_tree.csv"), header=None, na_rep="None")
    elif technique == "MLP":
        max_row.T.to_csv(os.path.join(root, "memory", "best_mlp.csv"), header=None, na_rep="None")
    else:
        print("Error: No valid classification technique")


# convert strings in dict into the suitable datatypes
def convert(params):
    for sub in params:
        if params[sub][0] in ["True", "False"]:
            params[sub][0] = bool(params[sub][0])
        elif params[sub][0].isdecimal():
            params[sub][0] = int(params[sub][0])
        elif params[sub][0].replace('.', '', 1).isdigit():
            params[sub][0] = float(params[sub][0])
        elif params[sub][0].find("(") != -1 and params[sub][0].find(")") != -1 and params[sub][0].find(",") != -1:
            params[sub][0] = eval(params[sub][0])
        elif params[sub][0] == "None":
            params[sub][0] = None

    # delete score key as it's not suitable for parameter space
    params = {x: params[x] for x in params if x not in ["score"]}

    return params
