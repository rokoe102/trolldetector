from sklearn import metrics
from prettytable import PrettyTable, ALL


# stores and prints the results of a custom classification
class CustomReport:
    def __init__(self, y_test, predicted):
        self.conf = metrics.confusion_matrix(y_test, predicted)

        self.accuracy = metrics.accuracy_score(y_test, predicted)

        self.precision = metrics.precision_score(y_test, predicted, pos_label="troll", zero_division=True)
        self.npv = metrics.precision_score(y_test, predicted, pos_label="nontroll", zero_division=True)

        self.recall = metrics.recall_score(y_test, predicted, pos_label="troll", zero_division=True)
        self.specifity = metrics.recall_score(y_test, predicted, pos_label="nontroll", zero_division=True)

        self.f1 = metrics.f1_score(y_test, predicted, pos_label="troll", zero_division=True)

    def print(self):
        print("+----------------------------------------------------+")
        print("|                      REPORT                        |")
        print("+----------------------------------------------------+")

        # extract values from confusion matrix

        tn, fp, fn, tp = self.conf.ravel()

        print("\n+----------------------------------------------------+")
        print("|                 confusion matrix                   |")
        print("+----------------------------------------------------+")

        matrix = PrettyTable(["", "labelled nontroll", "labelled troll", "total"])
        matrix.vrules = ALL
        matrix.hrules = ALL
        matrix.add_row(["true nontroll", tn, fp, tn+fp])
        matrix.add_row(["true troll", fn, tp, fn+tp])
        matrix.add_row(["total", tn+fn, fp+tp, tn+fp+fn+tp])

        print(matrix)

        print("\n+----------------------------------------------------+")
        print("|                performance metrics                 |")
        print("+----------------------------------------------------+")
        score_table = PrettyTable(["accuracy", "precision", "NPV", "recall", "specifity", "f1"])
        score_table.add_row(["%0.3f" % self.accuracy,
                             "%0.3f" % self.precision,
                             "%0.3f" % self.npv,
                             "%0.3f" % self.recall,
                             "%0.3f" % self.specifity,
                             "%0.3f" % self.f1
                             ])
        print(score_table)
