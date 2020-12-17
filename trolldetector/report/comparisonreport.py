import pandas as pd
from prettytable import PrettyTable, ALL


# stores and prints the results of the comparison of all classification techniques
class ComparisonReport:
    def __init__(self, results):

        # save score for every performance metric

        df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.accuracy = df

        df = pd.DataFrame(list(zip(results["mean_test_precision_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.precision = df

        df = pd.DataFrame(list(zip(results["mean_test_npv_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.npv = df

        df = pd.DataFrame(list(zip(results["mean_test_recall_score"].tolist(), results["params"])),
                          columns=["score", "params"])
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

        df = pd.DataFrame(list(zip(results["mean_fit_time"].tolist(), results["params"])), columns=["time", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.train_time = df

        df = pd.DataFrame(list(zip(results["mean_score_time"].tolist(), results["params"])), columns=["time", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.test_time = df

        self.combinations = results["params"]

    def print(self):

        print("+----------------------------------------------------+")
        print("|                      REPORT                        |")
        print("|----------------------------------------------------|")
        print("|                    mean scores                     |")
        print("+----------------------------------------------------+")

        # print the mean score in every performance metric for every classifier

        score_table = PrettyTable(["", "accuracy", "precision", "NPV", "recall", "specifity", "f1"])
        score_table.hrules = ALL

        for clf in self.accuracy["clf"].unique():
            clf_score = {"accuracy": self.accuracy[self.accuracy["clf"] == clf]["score"],
                         "precision": self.precision[self.precision["clf"] == clf]["score"],
                         "npv": self.npv[self.npv["clf"] == clf]["score"],
                         "recall": self.recall[self.recall["clf"] == clf]["score"],
                         "specifity": self.specifity[self.specifity["clf"] == clf]["score"],
                         "f1": self.f_one[self.f_one["clf"] == clf]["score"]
                         }

            # print without '()'
            name = str(clf)
            bracket = name.find("(")
            name = name[:bracket]

            score_table.add_row([name,
                                 "%0.3f" % clf_score["accuracy"].iat[0],
                                 "%0.3f" % clf_score["precision"].iat[0],
                                 "%0.3f" % clf_score["npv"].iat[0],
                                 "%0.3f" % clf_score["recall"].iat[0],
                                 "%0.3f" % clf_score["specifity"].iat[0],
                                 "%0.3f" % clf_score["f1"].iat[0]
                                 ])

        print(score_table)

        print("+----------------------------------------------------+")
        print("|                    mean runtime                    |")
        print("+----------------------------------------------------+")

        # print the mean runtime for every classifier

        runtime_table = PrettyTable(["", "train time (s)", "test time (s)", "total (s)"])
        runtime_table.hrules = ALL
        for clf in self.train_time["clf"].unique():
            runtime = {"train_time": self.train_time[self.train_time["clf"] == clf]["time"],
                       "test_time": self.test_time[self.test_time["clf"] == clf]["time"]
                       }

            name = str(clf)
            bracket = name.find("(")
            name = name[:bracket]

            runtime_table.add_row([name,
                                   round(runtime["train_time"].iat[0]),
                                   round(runtime["test_time"].iat[0]),
                                   round(runtime["train_time"].iat[0]) + round(runtime["test_time"].iat[0])
                                   ])

        print(runtime_table)
