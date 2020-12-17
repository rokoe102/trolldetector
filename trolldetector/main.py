import argparse
from trolldetector.KNN import knn
from trolldetector.MLP import mlp
from trolldetector.NB import nb
from trolldetector.SVM import svm
from trolldetector.all import all
from trolldetector.dtree import dtree
from trolldetector.parsing.commonarguments import CommonArguments
from trolldetector.parsing import check


def main():
    # define main parser
    pparser = argparse.ArgumentParser(prog="trolldetector")

    pparser._optionals.title = "Options"

    pparser.add_argument("--optimize", dest="optimize", action="store_true",
                         help="hyperparameter optimization (ignoring other options)")

    subparsers = pparser.add_subparsers(title="classification techniques", dest="command")


    # arguments all commands have in common
    pparser.add_argument("--ngram", dest="gram", metavar="<n>", type=check.ngram_warning, default=1,
                         help="changing the n-gram range to (1,n)")

    pparser.add_argument("--tfidf", dest="tfidf", action="store_true",
                         help="changes the feature weighting to TF-IDF")

    pparser.add_argument("--stopwords", dest="stop", action="store_true",
                         help="enables filtering of english stop words")

    pparser.add_argument("--dim", dest="dims", metavar="<components>", type=check.dim_warning, default=10,
                         help="determines the level of dimensionality reduction")

    pparser.add_argument("--test", dest="test_set", metavar="<(0,1)>", type=check.test, default=0.1,
                         help="changes the proportion of training and testing data")


    # parser for KNN command
    knn_parser = subparsers.add_parser("KNN", add_help=False, help="k-Nearest Neighbor Classification")

    knn_parser.add_argument("-k", dest="kvar", metavar="<n>", type=check.k_warning, default=5, help="changes k value")

    knn_parser.add_argument("-m", dest="metric", type=str, default="euclidean", choices=["euclidean", "manhattan"],
                            help="determines the metric for distance measurement")

    knn_parser.add_argument("-h", dest="help_knn", action="store_true", help="displays this help message")


    # parser for NB command
    nb_parser = subparsers.add_parser("NB", add_help=False, help="Naive Bayes Classification")

    nb_parser.add_argument("-d", "--dist", dest="distribution", type=str, default="gaussian",
                           choices=["gaussian", "multinomial", "complement"],
                           help="changes the presumed distribution of the data")

    nb_parser.add_argument("-h", dest="help_nb", action="store_true", help="displays this help message")


    # parser for SVM command
    svm_parser = subparsers.add_parser("SVM", add_help=False, help="Support-vector Machine Classification")

    svm_parser.add_argument("--cost", dest="cost", metavar="<(0,1]>", type=check.halfopen_interval, default=1.0,
                            help="changes the penalization parameter for misclassification")

    svm_parser.add_argument("-h", dest="help_svm", action="store_true", help="displays this help message")


    # parser for tree command
    tree_parser = subparsers.add_parser("tree", add_help=False, help="Decision Tree Classification")

    tree_parser.add_argument("-m", dest="splitting", type=str, default="gini", choices=["gini", "entropy"],
                             help="determines the metric used for finding the best split")

    tree_parser.add_argument("-h", dest="help_tree", action="store_true", help="displays this help message")


    # parser for MLP command
    mlp_parser = subparsers.add_parser("MLP", add_help=False, help="Multilayer Perceptron Classification")

    mlp_parser.add_argument("-a", dest="activation", type=str, default="relu", choices=["relu", "tanh", "logistic"],
                            help="changes the activation function")

    mlp_parser.add_argument("--iter", dest="iter", type=check.positive_integer, metavar="<n>", default=5,
                            help="stops training after n iterations with no improvement >= tol")

    mlp_parser.add_argument("--tol", dest="tol", type=check.tol_warning, metavar="<(0,1)>", default=0.0025,
                            help="determines tolerance for stopping condition")

    mlp_parser.add_argument("-h", dest="help_mlp", action="store_true", help="displays this help message")


    # parser for all command
    all_parser = subparsers.add_parser("all", add_help=False, help="compare all classificators")

    all_parser.add_argument("-h", dest="help_all", action="store_true", help="displays this help message")


    # get arguments
    args = pparser.parse_args()
    method = args.command

    # wrap the shared arguments
    cargs = CommonArguments(args.gram, args.tfidf, args.stop, args.dims, args.test_set)

    # perform hyperparameter optimization if "--optimize" is provided
    # otherwise: execute technique once with custom hyperparameters
    # (or print help message)

    if method == "KNN":
        if args.help_knn:
            knn_parser.print_help()

        elif args.optimize:
            knn.optimize()

        else:
            knn.train_and_test(args.kvar, args.metric, cargs)

    elif method == "NB":
        if args.help_nb:
            nb_parser.print_help()

        elif args.optimize:
            nb.optimize()

        else:
            nb.train_and_test(args.distribution, cargs)

    elif method == "SVM":
        if args.help_svm:
            svm_parser.print_help()

        elif args.optimize:
            svm.optimize()

        else:
            svm.train_and_test(args.cost, cargs)

    elif method == "tree":
        if args.help_tree:
            tree_parser.print_help()

        elif args.optimize:
            dtree.optimize()

        else:
            dtree.train_and_test(args.splitting, cargs)

    elif method == "MLP":
        if args.help_mlp:
            mlp_parser.print_help()

        elif args.optimize:
            mlp.optimize()

        else:
            mlp.train_and_test(args.activation, args.iter, args.tol, cargs)

    elif method == "all":
        if args.help_all:
            all_parser.print_help()

        else:
            all.compare()