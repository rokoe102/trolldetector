import argparse
from KNN import knn
from MLP import mlp
from NB import nb
from SVM import svm
from all import all
from dtree import dtree
from parsing.commonarguments import CommonArguments

#if __name__ == "__main__":
def main():

    # define main parser
    pparser = argparse.ArgumentParser(prog="trolldetector")
    pparser._optionals.title = "Options"
    pparser.add_argument("--optimize",dest="optimize", action="store_true", help="hyperparameter optimization (ignoring other options)")

    subparsers = pparser.add_subparsers(title="classification techniques", dest="command")

    # arguments all commands have in common
    pparser.add_argument("--ngram",dest="gram", metavar="<n>",type=int,default=1, help="changing the n-gram range to (1,n)")
    pparser.add_argument("--tfidf",dest="tfidf", action="store_true", help="changes the feature weighting to TF-IDF")
    pparser.add_argument("--stopwords", dest="stop", action="store_true", help="enables filtering of english stop words")
    pparser.add_argument("--dim", dest="dims",metavar="<components>", type=int,default=10, help="determines the level of dimensionality reduction")
    pparser.add_argument("-v", "--verbose", dest="verb", action="store_true", help="producing more detailed output")

    # parser for KNN command
    knn_parser = subparsers.add_parser("KNN", add_help=False, help="k-nearest neighbor classification")
    knn_parser.add_argument("-k", dest="kvar", metavar="<value>", type=int, default=5, help="changes k value")
    knn_parser.add_argument("-m", dest="metric", metavar="<metric>", type=str, default="euclidean", choices=["euclidean", "manhattan", "chebyshev"], help="determines the metric for distance measurement")
    knn_parser.add_argument("--test",dest="tpercKNN", metavar="<perc>", type=float,default=0.1, help="changes the proportion of test data")
    knn_parser.add_argument("-h", dest="helpKNN", action="store_true", help="displays this help message")

    # parser for NB command
    nb_parser = subparsers.add_parser("NB", add_help=False, help="Naive Bayes classification")
    nb_parser.add_argument("-d", "--distribution", dest="distNB", type=str, default="gaussian",choices = ["gaussian", "multinomial","bernoulli", "complement"], help="changes the presumed distribution of the data")
    nb_parser.add_argument("--test", dest="tpercNB", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    nb_parser.add_argument("-h", dest="helpNB", action="store_true", help="displays this help message")

    # parser for SVM command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="support-vector machine classification")
    svm_parser.add_argument("--cost", dest="cost", type=float, default=1.0, help="changes the penalization parameter for misclassification")
    svm_parser.add_argument("--test", dest="tpercSVM", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    svm_parser.add_argument("-h", dest="helpSVM", action="store_true", help="displays this help message")

    # parser for tree command
    tree_parser = subparsers.add_parser("tree", add_help=False, help="decision tree classification")
    tree_parser.add_argument("-m", dest="metrTree", type=str, default="gini", choices=["gini", "entropy"], help="determines the metric used for finding the best split")
    tree_parser.add_argument("--test", dest="tpercTree", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    tree_parser.add_argument("-h", dest="helpTree", action="store_true", help="displays this help message")

    # parser for MLP command
    mlp_parser = subparsers.add_parser("MLP", add_help=False, help="multi-layer perceptron classification")
    mlp_parser.add_argument("-a", dest="actMLP", type=str, default="relu", choices=["relu","tanh","identity","logistic"], help="changes the activation function")
    mlp_parser.add_argument("--iter", dest="iter", type=int, metavar="<n>", default=5, help="stops training after n iterations with no improvement >= tol")
    mlp_parser.add_argument("--tol", dest="tol", type=float, default=0.0025, help="determines tolerance for stopping condition")
    mlp_parser.add_argument("--test", dest="tpercMLP", metavar="<perc>", type=float, default=0.1, help="changes the proportion of test data")
    mlp_parser.add_argument("-h", dest="helpMLP", action="store_true", help="displays this help message")

    # parser for all command
    all_parser = subparsers.add_parser("all", add_help=False, help="compare all classificators")

    # get arguments
    args = pparser.parse_args()
    method = args.command

    # wrap the shared arguments
    cargs = CommonArguments(args.gram, args.tfidf, args.stop, args.dims, args.verb)

    # perform hyperparameter optimization if "--optimize" is provided
    # otherwise: execute technique once with custom hyperparameters

    if method == "KNN":
        if args.helpKNN:
            knn_parser.print_help()
        elif args.optimize:
            knn.optimize(args.tpercKNN, args.verb)
        else:
            knn.train_and_test(args.kvar, args.metric,  args.tpercKNN, cargs)

    elif method == "NB":
        if args.helpNB:
            nb_parser.print_help()
        elif args.optimize:
            nb.optimize(args.tpercNB, args.verb)
        else:
            nb.train_and_test(args.tpercNB, args.distNB, cargs)

    elif method == "SVM":
        if args.helpSVM:
            svm_parser.print_help()
        elif args.optimize:
            svm.optimize(args.tpercSVM, args.verb)
        else:
            svm.train_and_test(args.tpercSVM,  args.cost,cargs)

    elif method == "tree":
        if args.helpTree:
            tree_parser.print_help()
        elif args.optimize:
            dtree.optimize(args.tpercTree, args.verb)
        else:
            dtree.train_and_test(args.tpercTree,  args.metrTree, cargs)

    elif method == "MLP":
        if args.helpMLP:
            mlp_parser.print_help()
        elif args.optimize:
            mlp.optimize(args.tpercMLP, args.verb)
        else:
            mlp.train_and_test(args.actMLP,args.iter, args.tol,args.tpercMLP, cargs)

    elif method == "all":
        all.compare(args.verb)
