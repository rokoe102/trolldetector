import pandas as pd
import numpy as np


# noinspection PyStringFormat
class HypOptReport:
    def __init__(self, technique, results):
        self.technique = technique

        df = pd.DataFrame(list(zip(results["mean_test_accuracy_score"].tolist(), results["std_test_accuracy_score"].tolist(), results["params"])), columns=["score","std","params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.accuracy = df

        df = pd.DataFrame(list(zip(results["mean_test_precision_score"].tolist(), results["std_test_precision_score"].tolist(), results["params"])),
                          columns=["score","std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.precision = df

        df = pd.DataFrame(list(
            zip(results["mean_test_npv_score"].tolist(), results["std_test_precision_score"].tolist(),
                results["params"])),
                          columns=["score", "std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.npv = df

        df = pd.DataFrame(list(zip(results["mean_test_recall_score"].tolist(), results["std_test_recall_score"].tolist(), results["params"])),
                          columns=["score","std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.recall = df

        df = pd.DataFrame(list(
            zip(results["mean_test_specifity_score"].tolist(), results["std_test_precision_score"].tolist(),
                results["params"])),
                          columns=["score", "std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.specifity = df

        df = pd.DataFrame(list(zip(results["mean_test_f1_score"].tolist(), results["std_test_f1_score"], results["params"])),
                          columns=["score","std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.f_one = df

        self.combinations = results["params"]


    def print(self):
        self.print_common()
        if self.technique == "KNN":
            self.print_knn()
        elif self.technique == "NB":
            self.print_nb()
        elif self.technique == "SVM":
            self.print_svm()
        elif self.technique == "tree":
            self.print_dt()
        elif self.technique == "MLP":
            self.print_mlp()
        else:
            print("Error: no valid classification technique: " + self.technique)


    # print rankings for all combinations and mean scores for the
    # most common hyperparameters (TF/TF-IDF, n-grams, stop words)

    def print_common(self):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|                      REPORT                        |")
        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                  accuracy ranking                  |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 accuracy scores, std.deviation and params
        zipped = sorted(zip(self.accuracy["score"], self.accuracy["std"], self.combinations),key= lambda t: t[0], reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                 precision ranking                  |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 precision scores, std.deviation and params
        zipped = sorted(zip(self.precision["score"], self.precision["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                    NPV ranking                     |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 NPV scores, std.deviation and params
        zipped = sorted(zip(self.npv["score"], self.npv["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                   recall ranking                   |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 recall scores, std.deviation and params
        zipped = sorted(zip(self.recall["score"], self.recall["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                specifity ranking                   |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 specifity scores, std.deviation and params
        zipped = sorted(zip(self.specifity["score"], self.specifity["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                     f1 ranking                     |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        # print top 5 f1 scores, std.deviation and params
        zipped = sorted(zip(self.f_one["score"], self.f_one["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1


        # show mean scores for

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|            hyperparameters in detail               |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        avg_score = {"accuracy": np.mean(self.accuracy["score"]),
                     "precision": np.mean(self.precision["score"]),
                     "npv": np.mean(self.npv["score"]),
                     "recall": np.mean(self.recall["score"]),
                     "specifity": np.mean(self.specifity["score"]),
                     "f1": np.mean(self.f_one["score"])

                    }


        print("average scores")
        print("\taccuracy\tprecision\tNPV   \t\trecall\t\tspecifity\tf1")
        print("\t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f   \t%0.3f    \t%0.3f" % (
        avg_score["accuracy"], avg_score["precision"], avg_score["npv"], avg_score["recall"], avg_score["specifity"],
        avg_score["f1"]))


        print("              +-------------------+                   ")
        print("              | feature weighting |                   ")
        print("              +-------------------+                   ")

        tfidf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] == True]["score"]),
                       "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] == True]["score"]),
                       "npv": np.mean(self.npv[self.npv["tfidf__use_idf"] == True]["score"]),
                       "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] == True]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["tfidf__use_idf"] == True]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["tfidf__use_idf"] == True]["score"])
                      }
        tf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] == False]["score"]),
                    "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] == False]["score"]),
                    "npv": np.mean(self.npv[self.npv["tfidf__use_idf"] == False]["score"]),
                    "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] == False]["score"]),
                    "specifity": np.mean(self.specifity[self.specifity["tfidf__use_idf"] == False]["score"]),
                    "f1": np.mean(self.f_one[self.f_one["tfidf__use_idf"] == False]["score"])
                   }

        print("      \taccuracy\tprecision\tNPV   \t\trecall\t\tspecifity\tf1")
        print("TF    \t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f   \t%0.3f    \t%0.3f" % (tf_score["accuracy"], tf_score["precision"], tf_score["npv"], tf_score["recall"], tf_score["specifity"], tf_score["f1"]))
        print("TF-IDF\t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f   \t%0.3f    \t%0.3f" % (tfidf_score["accuracy"], tfidf_score["precision"], tfidf_score["npv"], tfidf_score["recall"], tfidf_score["specifity"], tfidf_score["f1"]))

        print("              +---------------------+                ")
        print("              | stop word filtering |                ")
        print("              +---------------------+                ")

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

        print("       \taccuracy\tprecision\tNPV   \t\trecall\t\tspecifity\tf1")
        print("english\t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f   \t%0.3f   \t%0.3f" % (stop_score["accuracy"], stop_score["precision"], stop_score["npv"], stop_score["recall"], stop_score["specifity"], stop_score["f1"]))
        print("none   \t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f   \t%0.3f   \t%0.3f" % (
                nostop_score["accuracy"], nostop_score["precision"], nostop_score["npv"], nostop_score["recall"], nostop_score["specifity"], nostop_score["f1"]))

        print("                +--------------+                    ")
        print("                | n-gram range |                    ")
        print("                +--------------+                    ")

        onegram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1,1)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1,1)]["score"]),
                         "npv": np.mean(self.npv[self.npv["vect__ngram_range"] == (1,1)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1,1)]["score"]),
                         "specifity": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 1)]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["vect__ngram_range"] == (1, 1)]["score"])
                        }
        twogram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1,2)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1,2)]["score"]),
                         "npv": np.mean(self.npv[self.npv["vect__ngram_range"] == (1,2)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1,2)]["score"]),
                         "specifity": np.mean(self.recall[self.recall["vect__ngram_range"] == (1, 2)]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["vect__ngram_range"] == (1, 2)]["score"])
                        }

        print("      \taccuracy\tprecision\tNPV   \t\trecall \t\tspecifity\tf1")
        print("(1,1) \t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f    \t%0.3f    \t%0.3f" % (onegram_score["accuracy"], onegram_score["precision"], onegram_score["npv"], onegram_score["recall"], onegram_score["specifity"], onegram_score["f1"]))
        print("(1,2) \t%0.3f   \t%0.3f    \t%0.3f   \t%0.3f    \t%0.3f    \t%0.3f" % (
                twogram_score["accuracy"], twogram_score["precision"], twogram_score["npv"], twogram_score["recall"], twogram_score["specifity"], twogram_score["f1"]))


    # print tables with mean scores of every KNN specific hyperparameter
    def print_knn(self):
        print("                 +---------+                        ")
        print("                 | k value |                        ")
        print("                 +---------+                        ")

        print("        \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for k in self.accuracy["clf__n_neighbors"].unique():
            k_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__n_neighbors"] == k]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__n_neighbors"] == k]["score"]),
                       "npv": np.mean(self.npv[self.npv["clf__n_neighbors"] == k]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__n_neighbors"] == k]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["clf__n_neighbors"] == k]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["clf__n_neighbors"] == k]["score"])
                      }
                      
            print("%-9s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % ("k = " + str(k), k_score["accuracy"], k_score["precision"], k_score["npv"], k_score["recall"], k_score["specifity"], k_score["f1"]))

        print("                 +---------+                        ")
        print("                 | metrics |                        ")
        print("                 +---------+                        ")

        print("         \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for metric in self.accuracy["clf__metric"].unique():
            metric_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__metric"] == metric]["score"]),
                            "precision": np.mean(self.precision[self.precision["clf__metric"] == metric]["score"]),
                            "npv": np.mean(self.npv[self.npv["clf__metric"] == metric]["score"]),
                            "recall": np.mean(self.recall[self.recall["clf__metric"] == metric]["score"]),
                            "specifity": np.mean(self.specifity[self.specifity["clf__metric"] == metric]["score"]),
                            "f1": np.mean(self.f_one[self.f_one["clf__metric"] == metric]["score"])
                            }

            print("%-10s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (metric, metric_score["accuracy"], metric_score["precision"], metric_score["npv"], metric_score["recall"], metric_score["specifity"], metric_score["f1"]))


    # print tables with mean scores of every Naive Bayes specific hyperparameter
    def print_nb(self):

        print("           +-----------------------+                 ")
        print("           | presumed distribution |                 ")
        print("           +-----------------------+                 ")

        print("        \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for dist in self.accuracy["clf"].unique():
            dist_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf"] == dist]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf"] == dist]["score"]),
                          "npv": np.mean(self.npv[self.npv["clf"] == dist]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf"] == dist]["score"]),
                          "specifity": np.mean(self.specifity[self.specifity["clf"] == dist]["score"]),
                          "f1": np.mean(self.f_one[self.f_one["clf"] == dist]["score"])
                         }

            print("%-15s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (
            dist, dist_score["accuracy"], dist_score["precision"], dist_score["npv"], dist_score["recall"], dist_score["specifity"], dist_score["f1"]))


    # print tables with mean scores of every SVM specific hyperparameter
    def print_svm(self):
        # accuracy ranking of C value
        print("                  +---------+                 ")
        print("                  | C value |                 ")
        print("                  +------ --+                 ")

        print("      \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for c in self.accuracy["clf__C"].unique():
            c_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__C"] == c]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__C"] == c]["score"]),
                       "npv": np.mean(self.npv[self.npv["clf__C"] == c]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__C"] == c]["score"]),
                       "specifity": np.mean(self.specifity[self.specifity["clf__C"] == c]["score"]),
                       "f1": np.mean(self.f_one[self.f_one["clf__C"] == c]["score"])
                       }

            print("%-6s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                str(c), c_score["accuracy"], c_score["precision"], c_score["npv"], c_score["recall"], c_score["specifity"], c_score["f1"]))


    # print tables with mean scores of every Decision Tree specific hyperparameter
    def print_dt(self):
        print("                  +-----------+                 ")
        print("                  | criterion |                 ")
        print("                  +-----------+                 ")

        print("       \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for criterion in self.accuracy["clf__criterion"].unique():
            crit_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__criterion"] == criterion]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf__criterion"] == criterion]["score"]),
                          "npv": np.mean(self.npv[self.npv["clf__criterion"] == criterion]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf__criterion"] == criterion]["score"]),
                          "specifity": np.mean(self.specifity[self.specifity["clf__criterion"] == criterion]["score"]),
                          "f1": np.mean(self.f_one[self.f_one["clf__criterion"] == criterion]["score"])
                         }

            print("%-7s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                criterion, crit_score["accuracy"], crit_score["precision"], crit_score["npv"], crit_score["recall"], crit_score["specifity"], crit_score["f1"]))


    # print tables with mean scores of every MLP specific hyperparameter
    def print_mlp(self):
        print("           +---------------------+                 ")
        print("           | activation function |                 ")
        print("           +---------------------+                 ")

        print("        \taccuracy\tprecision\tNPV\t\trecall\t\tspecifity\tf1")

        for activation in self.accuracy["clf__activation"].unique():
            act_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__activation"] == activation]["score"]),
                         "precision": np.mean(self.precision[self.precision["clf__activation"] == activation]["score"]),
                         "npv": np.mean(self.npv[self.npv["clf__activation"] == activation]["score"]),
                         "recall": np.mean(self.recall[self.recall["clf__activation"] == activation]["score"]),
                         "specifity": np.mean(self.specifity[self.specifity["clf__activation"] == activation]["score"]),
                         "f1": np.mean(self.f_one[self.f_one["clf__activation"] == activation]["score"])
                        }

            print("%-8s\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                activation, act_score["accuracy"], act_score["precision"], act_score["npv"], act_score["recall"], act_score["specifity"], act_score["f1"]))