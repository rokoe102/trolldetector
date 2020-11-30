from sklearn import metrics

# stores and prints the results of a custom classification
class CustomReport:
    def __init__(self, y_test, predicted):

        self.conf = metrics.confusion_matrix(y_test, predicted)
        self.clf_report = metrics.classification_report(y_test, predicted)

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
        print(self.clf_report)
