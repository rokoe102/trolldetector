import argparse
from KNN import knn

if __name__ == "__main__":
#def main():

    # define main parser
    pparser = argparse.ArgumentParser(prog="trolldetector")
    pparser._optionals.title = "Options"


    subparsers = pparser.add_subparsers(title="classification techniques", dest="command")

    # parser for knn command
    knn_parser = subparsers.add_parser("KNN", add_help=False, help="k-nearest neighbor classification")
    knn_parser.add_argument("-k", dest="kvar", metavar="<value>", type=int, default=5, help="changes k value")
    knn_parser.add_argument("-m", dest="metric", metavar="<metric>", type=str, default="euclidean", choices=["euclidean", "manhattan", "chebyshev"], help="determines the metric for distance measurement")
    knn_parser.add_argument("--test",dest="tperc", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    knn_parser.add_argument("-c", dest="comp", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    knn_parser.add_argument("-v", "--verbose", dest="verb", action="store_true", help="produces more detailed output")
    knn_parser.add_argument("-h", dest="help", action="store_true", help="displays this help message")

    # parser for svm command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="support-vector machine classification")

    # parser for nb command
    nb_parser = subparsers.add_parser("NB", add_help=False, help="Naive Bayes classification")

    # get arguments
    args = pparser.parse_args()
    method = args.command

    if method == "KNN":
        if(args.help == True):
            knn_parser.print_help()
        else:
            knn.classify(args.kvar, args.metric, args.tperc, args.comp, args.verb)

    elif method == "SVM":
        print("construction site -- come again later")

    elif method == "NB":
        print("construction site -- come again later")


