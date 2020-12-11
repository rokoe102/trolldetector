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

        tn, fp, fn, tp = self.conf.ravel()

        print("+----------------------------------------------------+")
        print("|                 confusion matrix                   |")
        print("+----------------------------------------------------+")

        print("\t\ttested nontroll\t\ttested troll")
        print("true nontroll\t%d\t\t\t%d" % (tn, fp))
        print("true troll\t%d\t\t\t%d" % (fn, tp))

        print("+----------------------------------------------------+")
        print("|                performance metrics                 |")
        print("+----------------------------------------------------+")
        score_table = PrettyTable(["accuracy", "precision", "npv", "recall", "specifity", "f1"])
        score_table.add_row([round(self.accuracy,3),
                             round(self.precision,3),
                             round(self.npv,3),
                             round(self.recall,3),
                             round(self.specifity,3),
                             round(self.f1,3)
                            ])
        print(score_table)

