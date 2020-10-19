import pandas as pd
import numpy as np

class HypOptReport:
    def __init__(self, technique, results):
        self.technique = technique

        df = pd.DataFrame(list(zip(results["mean_test_score"].tolist(), results["std_test_score"].tolist(), results["params"])), columns=["score","std", "params"])
        df = pd.concat([df.drop(["params"], axis=1), df["params"].apply(pd.Series)], axis=1)
        self.results = df

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
        print("|               ranking of combinations              |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        rank = 1

        zipped = sorted(zip(self.results["score"], self.results["std"], self.combinations),key= lambda t: t[0], reverse=True)
        for mean, std, params in zipped:
            print("[" + str(rank) + "]", end=" ")
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            rank += 1

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("|            hyperparameters in detail               |")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        tfidf_score = np.mean(self.results[self.results["tfidf__use_idf"] == True]["score"])
        tf_score = np.mean(self.results[self.results["tfidf__use_idf"] == False]["score"])

        print("              | feature weighting |                   ")
        print("              +-------------------+                   ")

        if tfidf_score > tf_score:
            print("[1] TF-IDF (mean %0.3f)" % tfidf_score)
            print("[2] TF (mean %0.3f)" % tf_score)
        elif tfidf_score < tf_score:
            print("[1] TF (mean %0.3f)" % tf_score)
            print("[2] TF-IDF (mean %0.3f)" % tfidf_score)

        else:
            print("[1] TF-IDF (mean %0.3f), TF (mean %0.3f)" % (tfidf_score,tf_score))

        stop_score = np.mean(self.results[self.results["vect__stop_words"] == "english"]["score"])
        nostop_score = np.mean(self.results[self.results["vect__stop_words"].isnull()]["score"])

        print("              +---------------------+                ")
        print("              | stop word filtering |                ")
        print("              +---------------------+                ")

        if stop_score > nostop_score:
            print("[1] english (mean %0.3f)" % stop_score)
            print("[2] none (mean %0.3f)" % nostop_score)

        elif stop_score < nostop_score:
            print("[1] none (mean %0.3f)" % nostop_score)
            print("[2] english (mean %0.3f)" % stop_score)

        else:
            print("[1] english (mean %0.3f), none (mean %0.3f)" % (stop_score, nostop_score))

        onegram_score = np.mean(self.results[self.results["vect__ngram_range"] == (1, 1)]["score"])
        twogram_score = np.mean(self.results[self.results["vect__ngram_range"] == (1, 2)]["score"])

        print("                +--------------+                    ")
        print("                | n-gram range |                    ")
        print("                +--------------+")

        if onegram_score > twogram_score:
            print("[1] 1-grams (mean %0.3f)" % onegram_score)
            print("[2] 1-grams + 2-grams (mean %0.3f)" % twogram_score)

        elif onegram_score < twogram_score:
            print("[1] 1-grams + 2 grams (mean %0.3f)" % twogram_score)
            print("[2] 1-grams (mean %0.3f)" % onegram_score)

        else:
            print("[1] 1-grams (mean %0.3f), 1-grams + 2-grams (mean %0.3f)" % (onegram_score, twogram_score))


    # print accuracy rankings for KNN specific hyperparameters

    def print_knn(self):
        # accuracy ranking of different k values
        print("                 +---------+                        ")
        print("                 | k value |                        ")
        print("                 +---------+                        ")

        k_scores = []
        for k in self.results["clf__n_neighbors"].unique():
            k_scores.append(np.mean(self.results[self.results["clf__n_neighbors"] == k]["score"]))

        rank = 1
        for score, k in sorted(zip(k_scores, self.results["clf__n_neighbors"].unique()), reverse=True):
            print("[" + str(rank) + "] k = " + str(k) + " (mean %0.3f)" % score)
            rank += 1

        # accuracy ranking of the three metrics
        print("                 +---------+                        ")
        print("                 | metrics |                        ")
        print("                 +---------+                        ")

        metric_scores = []
        for metric in self.results["clf__metric"].unique():
            metric_scores.append(np.mean(self.results[self.results["clf__metric"] == metric]["score"]))

        rank = 1
        for score, metric in sorted(zip(metric_scores, self.results["clf__metric"].unique()), reverse=True):
            print("[" + str(rank) + "] " + metric + " (mean %0.3f)" % score)
            rank += 1


    def print_nb(self):
        # accuracy ranking of different classificators
        # GaussianNB, MultinomialNB, ComplementNB
        print("           +-----------------------+                 ")
        print("           | presumed distribution |                 ")
        print("           +-----------------------+                 ")

        dist_scores = []
        for dist in self.results["clf"].unique():
            dist_scores.append(np.mean(self.results[self.results["clf"] == dist]["score"]))

        rank = 1

        for score, dist in sorted(zip(dist_scores, self.results["clf"].unique()), reverse=True):
            print("[" + str(rank) + "] " + str(dist) + " (mean %0.3f)" % score)
            rank += 1

    def print_svm(self):
        # accuracy ranking of C value
        print("                  +---------+                 ")
        print("                  | C value |                 ")
        print("                  +------ --+                 ")

        c_scores = []
        for c in self.results["clf__C"].unique():
            c_scores.append(np.mean(self.results[self.results["clf__C"] == c]["score"]))

        rank = 1
        for score, c in sorted(zip(c_scores, self.results["clf__C"].unique()), reverse=True):
            print("[" + str(rank) + "] " + str(c) + " (mean %0.3f)" % score)
            rank += 1

    def print_dt(self):
        print("                  +-----------+                 ")
        print("                  | criterion |                 ")
        print("                  +-----------+                 ")

        crit_scores = []
        for criterion in self.results["clf__criterion"].unique():
            crit_scores.append(np.mean(self.results[self.results["clf__criterion"] == criterion]["score"]))

        rank = 1
        for score, criterion in sorted(zip(crit_scores, self.results["clf__criterion"].unique()), reverse=True):
            print("[" + str(rank) + "] " + str(criterion) + " (mean %0.3f)" % score)
            rank += 1

    def print_mlp(self):
        print("           +---------------------+                 ")
        print("           | activation function |                 ")
        print("           +---------------------+                 ")

        act_scores = []
        for activation_function in self.results["clf__activation"].unique():
            act_scores.append(np.mean(self.results[self.results["clf__activation"] == activation_function]["score"]))

        rank = 1
        for score, activation_function in sorted(zip(act_scores, self.results["clf__activation"].unique()), reverse=True):
            print("[" + str(rank) + "] " + str(activation_function) + " (mean %0.3f)" % score)
            rank += 1