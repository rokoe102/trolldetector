import argparse

if __name__ == "__main__":

    # define main parser
    pparser = argparse.ArgumentParser(prog="trolldetector")
    pparser._optionals.title = "Options"


    subparsers = pparser.add_subparsers(title="classification techniques", dest="command")
    #subparsers.required = True

    # parser for knn command
    knn_parser = subparsers.add_parser("KNN", add_help=False, help="k-nearest neighbor classification")
    knn_parser.add_argument("-k", dest="kvar", metavar="<value>", type=int, help="changes k value")
    knn_parser.add_argument("-m", dest="metric", metavar="<metric>", type=str, help="determines the metric for distance measurement")
    knn_parser.add_argument("--training",dest="tperc", metavar="<perc>", type=int, help="changes the proportion of training data")


    # parser for svm command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="support-vector machine classification")

    args = pparser.parse_args()
    method = args.command

    if method == "KNN":
        # TO BE CONTINUED
        print("KNN")

    elif method == "SVM":
        print("construction site -- come again later")


