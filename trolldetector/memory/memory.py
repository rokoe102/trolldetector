import pandas as pd
import sklearn.neighbors
from sklearn import naive_bayes, neighbors, svm, tree, neural_network

# get a dict of hyperparameters from save file
def load(technique):
     dict = []
     if technique == "KNN":
         dict = pd.read_csv("memory/best_knn.csv",index_col=0, header=None).T.to_dict("list")
         dict = convert(dict)
         clf = getattr(neighbors, dict["clf"][0].replace("(","").replace(")", ""))
         dict["clf"] = [clf()]

     elif technique == "NB":
         dict = pd.read_csv("memory/best_nb.csv",index_col=0, header=None).T.to_dict("list")
         dict = convert(dict)
         clf = getattr(naive_bayes, dict["clf"][0].replace("(", "").replace(")", ""))
         dict["clf"] = [clf()]

     elif technique == "SVM":
         dict = pd.read_csv("memory/best_svm.csv",index_col=0, header=None).T.to_dict("list")
         dict = convert(dict)
         clf = getattr(svm, dict["clf"][0].replace("(", "").replace(")", ""))
         dict["clf"] = [clf()]

     elif technique == "tree":
         dict = pd.read_csv("memory/best_tree.csv",index_col=0, header=None).T.to_dict("list")
         dict = convert(dict)
         print(dict)
         clf = getattr(tree, dict["clf"][0].replace("(", "").replace(")", ""))
         dict["clf"] = [clf()]

     elif technique == "MLP":
         dict = pd.read_csv("memory/best_mlp.csv",index_col=0, header=None).T.to_dict("list")
         dict = convert(dict)
         clf = getattr(neural_network, dict["clf"][0].replace("(", "").replace(")", ""))
         dict["clf"] = [clf()]
     else:
         print("Error: No valid classification technique")

     return dict

# save best results of hyperparameter optimization as csv
def save(results, technique):
    df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(),results["params"])), columns=["score", "params"])
    df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)

    max_row = df[ df["score"] == max(df["score"])]

    if technique == "KNN":
        max_row.T.to_csv("memory/best_knn.csv", header=None, na_rep="None")
    elif technique == "NB":
        max_row.T.to_csv("memory/best_nb.csv", header=None, na_rep="None")
    elif technique == "SVM":
        max_row.T.to_csv("memory/best_svm.csv", header=None, na_rep="None")
    elif technique == "tree":
        max_row.T.to_csv("memory/best_tree.csv", header=None, na_rep="None")
    elif technique == "MLP":
        max_row.T.to_csv("memory/best_mlp.csv", header=None, na_rep="None")
    else:
        print("Error: No valid classification technique")

# convert strings in dict into the suitable datatypes
def convert(dict):
    for sub in dict:
        if dict[sub][0] in ["True", "False"]:
            dict[sub][0] = bool(dict[sub][0])
        elif dict[sub][0].isdecimal():
            dict[sub][0] = int(dict[sub][0])
        elif dict[sub][0].replace('.', '', 1).isdigit():
            dict[sub][0] = float(dict[sub][0])
        elif dict[sub][0].find("(") != -1 and dict[sub][0].find(")") != -1 and dict[sub][0].find(",") != -1:
            dict[sub][0] = eval(dict[sub][0])
        elif dict[sub][0] == "None":
            dict[sub][0] = None

    # delete score key as it's not suitable for parameter space
    dict = {x: dict[x] for x in dict if x not in ["score"]}

    return dict
