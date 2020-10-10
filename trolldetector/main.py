import argparse
from KNN import knn
from NB import nb
from SVM import svm
from dtree import dtree
from MLP import mlp
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
    knn_parser.add_argument("-c", dest="compKNN", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    knn_parser.add_argument("--test",dest="tpercKNN", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    knn_parser.add_argument("-v", "--verbose", dest="verbKNN", action="store_true", help="produces more detailed output")
    knn_parser.add_argument("-h", dest="helpKNN", action="store_true", help="displays this help message")

    # parser for nb command
    nb_parser = subparsers.add_parser("NB", add_help=False, help="Naive Bayes classification")
    nb_parser.add_argument("-d", "--distribution", dest="distNB", type=str, default="gaussian",choices = ["gaussian", "multinomial"], help="changes the presumed distribution of the data")
    nb_parser.add_argument("--tfidf", dest="tfidfNB", action="store_true", help="changes the feature weighting to TF-IDF")
    nb_parser.add_argument("-c", dest="compNB", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    nb_parser.add_argument("--test", dest="tpercNB", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    nb_parser.add_argument("-v", "--verbose", dest="verbNB", action="store_true", help="produces more detailed output")
    nb_parser.add_argument("-h", dest="helpNB", action="store_true", help="displays this help message")

    # parser for svm command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="support-vector machine classification")
    svm_parser.add_argument("--cost", dest="cost", type=float, default=1.0, help="changes the penalization parameter for misclassification")
    svm_parser.add_argument("--tfidf", dest="tfidfSVM", action="store_true", help="changes the feature weighting to TF-IDF")
    svm_parser.add_argument("-c", dest="compSVM", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    svm_parser.add_argument("--test", dest="tpercSVM", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    svm_parser.add_argument("-v", "--verbose", dest="verbSVM", action="store_true", help="produces more detailed output")
    svm_parser.add_argument("-h", dest="helpSVM", action="store_true", help="displays this help message")

    # parser for tree command
    tree_parser = subparsers.add_parser("tree", add_help=False, help="decision tree classification")
    tree_parser.add_argument("-m", dest="metrTree", type=str, default="gini", choices=["gini", "entropy"], help="determines the metric used for finding the best split")
    tree_parser.add_argument("--tfidf", dest="tfidfTree", action="store_true", help="changes the feature weighting to TF-IDF")
    tree_parser.add_argument("-c", dest="compTree", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    tree_parser.add_argument("--test", dest="tpercTree", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    tree_parser.add_argument("-v", "--verbose", dest="verbTree", action="store_true", help="produces more detailed output")
    tree_parser.add_argument("-h", dest="helpTree", action="store_true", help="displays this help message")

    # parser for MLP command
    mlp_parser = subparsers.add_parser("MLP", add_help=False, help="multi-layer perceptron classification")
    mlp_parser = subparsers.add_parser("-a", dest="actMLP", type=str, default="relu", choices=["relu","tanh","identity","logistic"], help="changes the activation function")
    mlp_parser.add_argument("--tfidf", dest="tfidfMLP", action="store_true", help="changes the feature weighting to TF-IDF")
    mlp_parser.add_argument("-c", dest="compMLP", metavar="<components>", type=int, default=5, help="changes the desired level of dimensionality reduction")
    mlp_parser.add_argument("--test", dest="tpercMLP", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    mlp_parser.add_argument("-v", "--verbose", dest="verbMLP", action="store_true", help="produces more detailed output")
    mlp_parser.add_argument("-h", dest="helpMLP", action="store_true", help="displays this help message")

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

    elif method == "tree":
        if args.helpTree == True:
            tree_parser.print_help()
        else:
            dtree.classify(args.tpercTree, args.compTree, args.tfidfTree, args.metrTree, args.verbTree)

    elif method == "MLP":
        if args.helpMLP:
            mlp_parser.print_help()
        else:
            mlp.classify(args.actMLP,args.tpercMLP, args.compMLP, args.tfidfMLP, args.verbMLP)

    # measure and print runtime
    runtime = time.process_time() - start
    minutes, seconds = divmod(runtime, 60)
    print("Runtime: ", end="")
    print("{:0>2}:{:05.2f}".format(int(minutes),seconds))