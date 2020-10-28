import pandas as pd
import numpy as np

class ComparisonReport:
    def __init__(self, results):
        df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(), results["params"])), columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.accuracy = df

        df = pd.DataFrame(list(zip(results["mean_test_precision_score"].tolist(), results["params"])),columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.precision = df

        df = pd.DataFrame(list(zip(results["mean_test_npv_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.npv = df


        df = pd.DataFrame(list(zip(results["mean_test_recall_score"].tolist(),results["params"])), columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.recall = df

        df = pd.DataFrame(list(zip(results["mean_test_specifity_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.specifity = df

        df = pd.DataFrame(list(zip(results["mean_test_f1_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.f_one = df

        df = pd.DataFrame(list(zip(results["mean_fit_time"].tolist(),results["params"])),columns=["time", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.train_time = df

        df = pd.DataFrame(list(zip(results["mean_score_time"].tolist(), results["params"])), columns=["time", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.test_time = df



        self.combinations = results["params"]

    def print(self):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|                      REPORT                        |")
        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                    mean scores                     |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")


        print("                      \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for clf in self.accuracy["clf"].unique():
            clf_score = {"accuracy": self.accuracy[self.accuracy["clf"] == clf]["score"],
                         "precision": self.precision[self.precision["clf"] == clf]["score"],
                         "npv": self.npv[self.npv["clf"] == clf]["score"],
                         "recall": self.recall[self.recall["clf"] == clf]["score"],
                         "specifity": self.specifity[self.specifity["clf"] == clf]["score"],
                         "f1": self.f_one[self.f_one["clf"] == clf]["score"]
                        }

            name = str(clf)
            bracket = name.find("(")

            print("%-23s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                   name[:bracket], clf_score["accuracy"], clf_score["precision"], clf_score["npv"], clf_score["recall"], clf_score["specifity"], clf_score["f1"]))

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|                    mean runtime                    |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        print("                      \ttrain time (s)\ttest time (s)\ttotal")
        for clf in self.train_time["clf"].unique():
            runtime = {"train_time": self.train_time[self.train_time["clf"] == clf]["time"],
                       "test_time": self.test_time[self.test_time["clf"] == clf]["time"]
                      }

            name = str(clf)
            bracket = name.find("(")

            print("%-23s\t%0.2f\t\t%0.2f\t\t%0.2f" % (name[:bracket], runtime["train_time"], runtime["test_time"],
                                         runtime["train_time"] + runtime["test_time"]))