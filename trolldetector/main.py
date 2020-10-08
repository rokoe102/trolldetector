import argparse
from KNN import knn
from NB import nb
from SVM import svm
import time

if __name__ == "__main__":
#def main():

    start = time.process_time()

    # define main parser
    pparser = argparse.ArgumentParser(prog="trolldetector")
    pparser._optionals.title = "Options"


    subparsers = pparser.add_subparsers(title="classification techniques", dest="command")

    # parser for knn command
    knn_parser = subparsers.add_parser("KNN", add_help=False, help="k-nearest neighbor classification")
    knn_parser.add_argument("-k", dest="kvar", metavar="<value>", type=int, default=5, help="changes k value")
    knn_parser.add_argument("-m", dest="metric", metavar="<metric>", type=str, default="euclidean", choices=["euclidean", "manhattan", "chebyshev"], help="determines the metric for distance measurement")
    knn_parser.add_argument("--tf", dest="tfKNN", action="store_true", help="changes the feature weighting to TF")
    knn_parser.add_argument("--test",dest="tpercKNN", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    knn_parser.add_argument("-c", dest="compKNN", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    knn_parser.add_argument("-v", "--verbose", dest="verbKNN", action="store_true", help="produces more detailed output")
    knn_parser.add_argument("-h", dest="helpKNN", action="store_true", help="displays this help message")

    # parser for nb command
    nb_parser = subparsers.add_parser("NB", add_help=False, help="Naive Bayes classification")
    nb_parser.add_argument("--test",dest="tpercNB", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    nb_parser.add_argument("-d", "--distribution", dest="distNB", type=str, default="gaussian",help="changes the presumed distribution of the data")
    nb_parser.add_argument("-c", dest="compNB", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    nb_parser.add_argument("--tfidf", dest="tfidfNB", action="store_true", help="changes the feature weighting to TF-IDF")
    nb_parser.add_argument("-v", "--verbose", dest="verbNB", action="store_true", help="produces more detailed output")
    nb_parser.add_argument("-h", dest="helpNB", action="store_true", help="displays this help message")

    # parser for svm command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="support-vector machine classification")
    svm_parser.add_argument("--test",dest="tpercSVM", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    svm_parser.add_argument("-c", dest="compSVM", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    svm_parser.add_argument("--tfidf", dest="tfidfSVM", action="store_true", help="changes the feature weighting to TF-IDF")
    svm_parser.add_argument("--cost", dest="cost", type=float, default=1.0, help="changes the penalization parameter for misclassification")
    svm_parser.add_argument("-v", "--verbose", dest="verbSVM", action="store_true", help="produces more detailed output")
    svm_parser.add_argument("-h", dest="helpSVM", action="store_true", help="displays this help message")

    # get arguments
    args = pparser.parse_args()
    method = args.command

    if method == "KNN":
        if args.helpKNN == True:
            knn_parser.print_help()
        else:
            knn.classify(args.kvar, args.metric, args.tfKNN, args.distNB, args.tpercKNN, args.compKNN, args.verb)


    elif method == "NB":
        if args.helpNB == True:
            nb_parser.print_help()
        else:
            nb.classify(args.tpercNB, args.compNB, args.tfidfNB,args.distNB, args.verbNB)

    elif method == "SVM":
        if args.helpSVM == True:
            svm_parser.print_help()
        else:
            svm.classify(args.tpercSVM, args.compSVM, args.tfidfSVM, args.cost, args.verbSVM)

    runtime = time.process_time() - start
    minutes, seconds = divmod(runtime, 60)
    print("Runtime: ", end="")
    print("{:0>2}:{:05.2f}".format(int(minutes),seconds))