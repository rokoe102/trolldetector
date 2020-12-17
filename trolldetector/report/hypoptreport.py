import pandas as pd
import numpy as np
from prettytable import PrettyTable, ALL


# stores and prints the results of an executed hyperparameter optimization
class HypOptReport:
    def __init__(self, technique, results):
        self.technique = technique

        # store metric score and the params for every performance metric

        df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.accuracy = df

        df = pd.DataFrame(list(zip(results["mean_test_precision_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.precision = df

        df = pd.DataFrame(list(
            zip(results["mean_test_npv_score"].tolist(),
                results["params"])),
            columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.npv = df

        df = pd.DataFrame(list(zip(results["mean_test_recall_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.recall = df

        df = pd.DataFrame(list(
            zip(results["mean_test_specifity_score"].tolist(),
                results["params"])),
            columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.specifity = df

        df = pd.DataFrame(list(zip(results["mean_test_f1_score"].tolist(), results["params"])),
                          columns=["score", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.f_one = df

        self.combinations = results["params"]

        self.train_time = results["mean_fit_time"].tolist()

        self.test_time = results["mean_score_time"].tolist()

    # print the results for the technique specific hyperparams
    def print(self):
        print("+----------------------------------------------------+")
        print("|                      REPORT                        |")
        print("+----------------------------------------------------+")

        # print best tuple
        self.print_best()

        table = PrettyTable(["", "accuracy", "precision", "NPV", "recall", "specifity", "f1"])
        table.hrules = ALL

        table = self.add_common(table)

        if self.technique == "KNN":
            table = self.print_knn(table)
        elif self.technique == "NB":
            table = self.print_nb(table)
        elif self.technique == "SVM":
            table = self.print_svm(table)
        elif self.technique == "tree":
            table = self.print_dt(table)
        elif self.technique == "MLP":
            table = self.print_mlp(table)
        else:
            print("Error: no valid classification technique: " + self.technique)

        print(table)

        self.print_runtime()

    # print the hyperparameters with the best accuracy
    def print_best(self):
        print("+----------------------------------------------------+")
        print("|                  best tuple                        |")
        print("+----------------------------------------------------+")

        best = self.accuracy[self.accuracy["score"] == max(self.accuracy["score"])]
        combination = best.to_dict("records")
        combination = combination[0]

        print("best accuracy: %0.3f" % round(combination["score"], 3))
        print("best tuple:", end=" ")
        combination = {x: combination[x] for x in combination if x not in ["score"]}
        print(combination)

    # print mean scores for the most common hyperparameters (TF/TF-IDF, n-grams, stop words)
    def add_common(self, table):

        # show mean scores for shared hyperparameters
        print("+----------------------------------------------------+")
        print("|            hyperparameters in detail               |")
        print("+----------------------------------------------------+")

        avg_score = {"accuracy": np.mean(self.accuracy["score"]),
                     "precision": np.mean(self.precision["score"]),
                     "npv": np.mean(self.npv["score"]),
                     "recall": np.mean(self.recall["score"]),
                     "specifity": np.mean(self.specifity["score"]),
                     "f1": np.mean(self.f_one["score"])

                     }

        table.add_row(["average",
                       "%0.3f" % avg_score["accuracy"],
                       "%0.3f" % avg_score["precision"],
                       "%0.3f" % avg_score["npv"],
                       "%0.3f" % avg_score["recall"],
                       "%0.3f" % avg_score["specifity"],
                       "%0.3f" % avg_score["f1"]
                       ])

        # determine mean performance for every specific hyperparameter

        tfidf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] is True]["score"]),
                       "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] is True]["score"]),
                       "npv": np.mean(self.npv[self.npv["tfidf__use_idf"] is True]["score"]),
                       "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] is True]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["tfidf__use_idf"] is True]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["tfidf__use_idf"] is True]["score"])
                       }
        tf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] is False]["score"]),
                    "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] is False]["score"]),
                    "npv": np.mean(self.npv[self.npv["tfidf__use_idf"] is False]["score"]),
                    "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] is False]["score"]),
                    "specifity": np.mean(self.specifity[self.specifity["tfidf__use_idf"] is False]["score"]),
                    "f1": np.mean(self.f_one[self.f_one["tfidf__use_idf"] is False]["score"])
                    }

        table.add_row(["TF",
                       "%0.3f" % tf_score["accuracy"],
                       "%0.3f" % tf_score["precision"],
                       "%0.3f" % tf_score["npv"],
                       "%0.3f" % tf_score["recall"],
                       "%0.3f" % tf_score["specifity"],
                       "%0.3f" % tf_score["f1"],
                       ])

        table.add_row(["TF-IDF",
                       "%0.3f" % tfidf_score["accuracy"],
                       "%0.3f" % tfidf_score["precision"],
                       "%0.3f" % tfidf_score["npv"],
                       "%0.3f" % tfidf_score["recall"],
                       "%0.3f" % tfidf_score["specifity"],
                       "%0.3f" % tfidf_score["f1"]
                       ])

        stop_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__stop_words"] == "english"]["score"]),
                      "precision": np.mean(self.precision[self.precision["vect__stop_words"] == "english"]["score"]),
                      "npv": np.mean(self.npv[self.npv["vect__stop_words"] == "english"]["score"]),
                      "recall": np.mean(self.recall[self.recall["vect__stop_words"] == "english"]["score"]),
                      "specifity": np.mean(self.specifity[self.specifity["vect__stop_words"] == "english"]["score"]),
                      "f1": np.mean(self.f_one[self.f_one["vect__stop_words"] == "english"]["score"])
                      }
        nostop_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__stop_words"].isnull()]["score"]),
                        "precision": np.mean(self.precision[self.precision["vect__stop_words"].isnull()]["score"]),
                        "npv": np.mean(self.npv[self.npv["vect__stop_words"].isnull()]["score"]),
                        "recall": np.mean(self.recall[self.recall["vect__stop_words"].isnull()]["score"]),
                        "specifity": np.mean(self.specifity[self.specifity["vect__stop_words"].isnull()]["score"]),
                        "f1": np.mean(self.f_one[self.f_one["vect__stop_words"].isnull()]["score"])
                        }

        table.add_row(["english stopwords",
                       "%0.3f" % stop_score["accuracy"],
                       "%0.3f" % stop_score["precision"],
                       "%0.3f" % stop_score["npv"],
                       "%0.3f" % stop_score["recall"],
                       "%0.3f" % stop_score["specifity"],
                       "%0.3f" % stop_score["f1"]
                       ])

        table.add_row(["no filtering",
                       "%0.3f" % nostop_score["accuracy"],
                       "%0.3f" % nostop_score["precision"],
                       "%0.3f" % nostop_score["npv"],
                       "%0.3f" % nostop_score["recall"],
                       "%0.3f" % nostop_score["specifity"],
                       "%0.3f" % nostop_score["f1"]
                       ])

        onegram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1, 1)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1, 1)]["score"]),
                         "npv": np.mean(self.npv[self.npv["vect__ngram_range"] == (1, 1)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 1)]["score"]),
                         "specifity": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 1)]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["vect__ngram_range"] == (1, 1)]["score"])
                         }
        twogram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1, 2)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1, 2)]["score"]),
                         "npv": np.mean(self.npv[self.npv["vect__ngram_range"] == (1, 2)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 2)]["score"]),
                         "specifity": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 2)]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["vect__ngram_range"] == (1, 2)]["score"])
                         }

        table.add_row(["1-grams",
                       "%0.3f" % onegram_score["accuracy"],
                       "%0.3f" % onegram_score["precision"],
                       "%0.3f" % onegram_score["npv"],
                       "%0.3f" % onegram_score["recall"],
                       "%0.3f" % onegram_score["specifity"],
                       "%0.3f" % onegram_score["f1"]
                       ])

        table.add_row(["1+2-grams",
                       "%0.3f" % twogram_score["accuracy"],
                       "%0.3f" % twogram_score["precision"],
                       "%0.3f" % twogram_score["npv"],
                       "%0.3f" % twogram_score["recall"],
                       "%0.3f" % twogram_score["specifity"],
                       "%0.3f" % twogram_score["f1"]
                       ])

        return table

    # print tables with mean scores of every KNN specific hyperparameter
    def print_knn(self, table):

        for k in self.accuracy["clf__n_neighbors"].unique():
            k_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__n_neighbors"] == k]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__n_neighbors"] == k]["score"]),
                       "npv": np.mean(self.npv[self.npv["clf__n_neighbors"] == k]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__n_neighbors"] == k]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["clf__n_neighbors"] == k]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["clf__n_neighbors"] == k]["score"])
                       }
            table.add_row(["k = " + str(k),
                           "%0.3f" % k_score["accuracy"],
                           "%0.3f" % k_score["precision"],
                           "%0.3f" % k_score["npv"],
                           "%0.3f" % k_score["recall"],
                           "%0.3f" % k_score["specifity"],
                           "%0.3f" % k_score["f1"]
                           ])

        for metric in self.accuracy["clf__metric"].unique():
            metric_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__metric"] == metric]["score"]),
                            "precision": np.mean(self.precision[self.precision["clf__metric"] == metric]["score"]),
                            "npv": np.mean(self.npv[self.npv["clf__metric"] == metric]["score"]),
                            "recall": np.mean(self.recall[self.recall["clf__metric"] == metric]["score"]),
                            "specifity": np.mean(self.specifity[self.specifity["clf__metric"] == metric]["score"]),
                            "f1": np.mean(self.f_one[self.f_one["clf__metric"] == metric]["score"])
                            }

            table.add_row([metric,
                           "%0.3f" % metric_score["accuracy"],
                           "%0.3f" % metric_score["precision"],
                           "%0.3f" % metric_score["npv"],
                           "%0.3f" % metric_score["recall"],
                           "%0.3f" % metric_score["specifity"],
                           "%0.3f" % metric_score["f1"]
                           ])
        return table

    # print tables with mean scores of every Naive Bayes specific hyperparameter
    def print_nb(self, table):

        for dist in self.accuracy["clf"].unique():
            dist_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf"] == dist]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf"] == dist]["score"]),
                          "npv": np.mean(self.npv[self.npv["clf"] == dist]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf"] == dist]["score"]),
                          "specifity": np.mean(self.specifity[self.specifity["clf"] == dist]["score"]),
                          "f1": np.mean(self.f_one[self.f_one["clf"] == dist]["score"])
                          }

            table.add_row([dist,
                           "%0.3f" % dist_score["accuracy"],
                           "%0.3f" % dist_score["precision"],
                           "%0.3f" % dist_score["npv"],
                           "%0.3f" % dist_score["recall"],
                           "%0.3f" % dist_score["specifity"],
                           "%0.3f" % dist_score["f1"]
                           ])
        return table

    # print tables with mean scores of every SVM specific hyperparameter
    # noinspection PyStringFormat,PyStringFormat,PyStringFormat,PyStringFormat,PyStringFormat,PyStringFormat
    def print_svm(self, table):

        for c in self.accuracy["clf__C"].unique():
            c_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__C"] == c]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__C"] == c]["score"]),
                       "npv": np.mean(self.npv[self.npv["clf__C"] == c]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__C"] == c]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["clf__C"] == c]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["clf__C"] == c]["score"])
                       }

            table.add_row(["C = " + str(c),
                           "%0.3f" % c_score["accuracy"],
                           "%0.3f" % c_score["precision"],
                           "%0.3f" % c_score["npv"],
                           "%0.3f" % c_score["recall"],
                           "%0.3f" % c_score["specifity"],
                           "%0.3f" % c_score["f1"]
                           ])
        return table

    # print tables with mean scores of every Decision Tree specific hyperparameter
    def print_dt(self, table):

        for criterion in self.accuracy["clf__criterion"].unique():
            crit_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__criterion"] == criterion]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf__criterion"] == criterion]["score"]),
                          "npv": np.mean(self.npv[self.npv["clf__criterion"] == criterion]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf__criterion"] == criterion]["score"]),
                          "specifity": np.mean(self.specifity[self.specifity["clf__criterion"] == criterion]["score"]),
                          "f1": np.mean(self.f_one[self.f_one["clf__criterion"] == criterion]["score"])
                          }

            table.add_row([criterion,
                           "%0.3f" % crit_score["accuracy"],
                           "%0.3f" % crit_score["precision"],
                           "%0.3f" % crit_score["npv"],
                           "%0.3f" % crit_score["recall"],
                           "%0.3f" % crit_score["specifity"],
                           "%0.3f" % crit_score["f1"]
                           ])
        return table

    # print tables with mean scores of every MLP specific hyperparameter
    def print_mlp(self, table):

        for activation in self.accuracy["clf__activation"].unique():
            act_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__activation"] == activation]["score"]),
                         "precision": np.mean(self.precision[self.precision["clf__activation"] == activation]["score"]),
                         "npv": np.mean(self.npv[self.npv["clf__activation"] == activation]["score"]),
                         "recall": np.mean(self.recall[self.recall["clf__activation"] == activation]["score"]),
                         "specifity": np.mean(self.specifity[self.specifity["clf__activation"] == activation]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["clf__activation"] == activation]["score"])
                         }

            table.add_row([activation,
                           "%0.3f" % act_score["accuracy"],
                           "%0.3f" % act_score["precision"],
                           "%0.3f" % act_score["npv"],
                           "%0.3f" % act_score["recall"],
                           "%0.3f" % act_score["specifity"],
                           "%0.3f" % act_score["f1"]
                           ])
        return table

    def print_runtime(self):
        time_table = PrettyTable()
        time_table.header = False
        time_table.hrules = ALL
        time_table.vrules = ALL
        time_table.add_row(["mean train time", str(round(np.mean(self.train_time))) + " seconds"])
        time_table.add_row(["mean test time", str(round(np.mean(self.test_time))) + " seconds"])

        print(time_table)
