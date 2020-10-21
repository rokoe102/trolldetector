import pandas as pd
import numpy as np

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

        df = pd.DataFrame(list(zip(results["mean_test_recall_score"].tolist(), results["std_test_recall_score"].tolist(), results["params"])),
                          columns=["score","std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.recall = df

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

    # noinspection PyStringFormat
    def print_common(self):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|                      REPORT                        |")
        print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
        print("|                  accuracy ranking                  |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

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
        print("|                   recall ranking                   |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1
        zipped = sorted(zip(self.recall["score"], self.recall["std"], self.combinations), key=lambda t: t[0],
                        reverse=True)
        for mean, std, params in zipped:
            if rank == 6:
                break
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|            hyperparameters in detail               |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("              | feature weighting |                   ")
        print("              +-------------------+                   ")

        tfidf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] == True]["score"]),
                       "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] == True]["score"]),
                       "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] == True]["score"]),
                      }
        tf_score = {"accuracy": np.mean(self.accuracy[self.accuracy["tfidf__use_idf"] == False]["score"]),
                    "precision": np.mean(self.precision[self.precision["tfidf__use_idf"] == False]["score"]),
                    "recall": np.mean(self.recall[self.recall["tfidf__use_idf"] == False]["score"]),
                   }

        print("      \taccuracy\tprecision\trecall")
        print("TF    \t%0.3f   \t%0.3f   \t%0.3f" % (tf_score["accuracy"], tf_score["precision"], tf_score["recall"]))
        print("TF-IDF\t%0.3f   \t%0.3f   \t%0.3f" % (tfidf_score["accuracy"], tfidf_score["precision"], tfidf_score["recall"]))

        print("              +---------------------+                ")
        print("              | stop word filtering |                ")
        print("              +---------------------+                ")

        stop_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__stop_words"] == "english"]["score"]),
                       "precision": np.mean(self.precision[self.precision["vect__stop_words"] == "english"]["score"]),
                       "recall": np.mean(self.recall[self.recall["vect__stop_words"] == "english"]["score"]),
                       }
        nostop_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__stop_words"].isnull()]["score"]),
                        "precision": np.mean(self.precision[self.precision["vect__stop_words"].isnull()]["score"]),
                        "recall": np.mean(self.recall[self.recall["vect__stop_words"].isnull()]["score"]),
                       }

        print("       \taccuracy\tprecision\trecall")
        print("english\t%0.3f   \t%0.3f   \t%0.3f" % (stop_score["accuracy"], stop_score["precision"], stop_score["recall"]))
        print("none   \t%0.3f   \t%0.3f   \t%0.3f" % (
                nostop_score["accuracy"], nostop_score["precision"], nostop_score["recall"]))

        print("                +--------------+                    ")
        print("                | n-gram range |                    ")
        print("                +--------------+                    ")

        onegram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1,1)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1,1)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1,1)]["score"]),
                        }
        twogram_score = {"accuracy": np.mean(self.accuracy[self.accuracy["vect__ngram_range"] == (1,2)]["score"]),
                         "precision": np.mean(self.precision[self.precision["vect__ngram_range"] == (1,2)]["score"]),
                         "recall": np.mean(self.recall[self.recall["vect__ngram_range"] == (1,2)]["score"]),
                        }

        print("      \taccuracy\tprecision\trecall")
        print("(1,1) \t%0.3f   \t%0.3f   \t%0.3f" % (onegram_score["accuracy"], onegram_score["precision"], onegram_score["recall"]))
        print("(1,2) \t%0.3f   \t%0.3f   \t%0.3f" % (
                twogram_score["accuracy"], twogram_score["precision"], twogram_score["recall"]))


    # print accuracy rankings for KNN specific hyperparameters
    def print_knn(self):
        # accuracy ranking of different k values
        print("                 +---------+                        ")
        print("                 | k value |                        ")
        print("                 +---------+                        ")

        print("         \taccuracy\tprecision\trecall")

        for k in self.accuracy["clf__n_neighbors"].unique():
            k_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__n_neighbors"] == k]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__n_neighbors"] == k]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__n_neighbors"] == k]["score"]),
                      }
                      
            print("%-9s\t%0.3f\t\t%0.3f\t\t%0.3f" % ("k = " + str(k), k_score["accuracy"], k_score["precision"], k_score["recall"]))

        # scoring of the three metrics
        print("                 +---------+                        ")
        print("                 | metrics |                        ")
        print("                 +---------+                        ")

        print("          \taccuracy\tprecision\trecall")

        for metric in self.accuracy["clf__metric"].unique():
            metric_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__metric"] == k]["score"]),
                            "precision": np.mean(self.precision[self.precision["clf__metric"] == k]["score"]),
                            "recall": np.mean(self.recall[self.recall["clf__metric"] == k]["score"]),
                            }

            print("%-10s\t%0.3f\t\t%0.3f\t\t%0.3f" % (metric, k_score["accuracy"], k_score["precision"], k_score["recall"]))


    def print_nb(self):
        # accuracy ranking of different classificators
        # GaussianNB, MultinomialNB, ComplementNB
        print("           +-----------------------+                 ")
        print("           | presumed distribution |                 ")
        print("           +-----------------------+                 ")

        print("               \taccuracy\tprecision\trecall")

        for dist in self.accuracy["clf"].unique():
            dist_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf"] == dist]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf"] == dist]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf"] == dist]["score"])
                         }


            print("%-15s\t%0.3f\t\t%0.3f\t\t%0.3f" % (
            dist, dist_score["accuracy"], dist_score["precision"], dist_score["recall"]))


    def print_svm(self):
        # accuracy ranking of C value
        print("                  +---------+                 ")
        print("                  | C value |                 ")
        print("                  +------ --+                 ")

        print("      \taccuracy\tprecision\trecall")

        for c in self.accuracy["clf__C"].unique():
            c_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__C"] == c]["score"]),
                          "precision": np.mean(self.precision[self.precision["clf__C"] == c]["score"]),
                          "recall": np.mean(self.recall[self.recall["clf__C"] == c]["score"])
                          }

            print("%-6s\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                str(c), c_score["accuracy"], c_score["precision"], c_score["recall"]))



    def print_dt(self):
        print("                  +-----------+                 ")
        print("                  | criterion |                 ")
        print("                  +-----------+                 ")

        print("       \taccuracy\tprecision\trecall")

        for criterion in self.accuracy["clf__criterion"].unique():
            crit_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__criterion"] == criterion]["score"]),
                       "precision": np.mean(self.precision[self.precision["clf__criterion"] == criterion]["score"]),
                       "recall": np.mean(self.recall[self.recall["clf__criterion"] == criterion]["score"])
                       }

            print("%-7s\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                criterion, crit_score["accuracy"], crit_score["precision"], crit_score["recall"]))

    def print_mlp(self):
        print("           +---------------------+                 ")
        print("           | activation function |                 ")
        print("           +---------------------+                 ")

        print("        \taccuracy\tprecision\trecall")

        for activation in self.accuracy["clf__activation"].unique():
            act_score = {"accuracy": np.mean(self.accuracy[self.accuracy["clf__activation"] == activation]["score"]),
                         "precision": np.mean(self.precision[self.precision["clf__activation"] == activation]["score"]),
                         "recall": np.mean(self.recall[self.recall["clf__activation"] == activation]["score"])
                        }

            print("%-8s\t%0.3f\t\t%0.3f\t\t%0.3f" % (
                activation, act_score["accuracy"], act_score["precision"], act_score["recall"]))