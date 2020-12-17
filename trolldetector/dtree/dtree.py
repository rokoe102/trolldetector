from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..memory import memory
from ..parsing import prepare
from ..report.hypoptreport import HypOptReport
from ..report.customreport import CustomReport

from prettytable import PrettyTable, ALL


# use the decision tree classification with custom hyperparameters
def train_and_test(metr, cargs):
    # print a summary of the selected arguments
    print_summary(metr, cargs)

    print("> loading datasets")

    # load and label datasets
    tweets, n_troll, n_nontroll = prepare.prepare_datasets()

    count_vec = CountVectorizer(ngram_range=(1, cargs.ngram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and weighting
    print("> preprocessing")

    X_train_counts = count_vec.fit_transform(tweets)
    if cargs.tfidf:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)

    # dimensionality reduction
    print("> reducing dimensions")

    svd = TruncatedSVD(n_components=cargs.dims, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)

    # splitting into training data and testing data
    print("> splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, prepare.get_target(n_troll, n_nontroll),
                                                        test_size=cargs.test_set, random_state=42, shuffle=True)

    treeClf = DecisionTreeClassifier(criterion=metr)

    # training
    print("> training the model")
    treeClf.fit(X_train, y_train)

    # testing
    print("> making predictions")
    predicted = treeClf.predict(X_test)

    # report the results
    report = CustomReport(y_test, predicted)
    report.print()


# performing a hyperparameter optimization for the decision tree classification
def optimize():
    print("+---------------------------------------------------------------+")
    print("| hyperparameter optimization for: decision tree classification |")
    print("+---------------------------------------------------------------+")

    tweets, n_troll, n_nontroll = prepare.prepare_datasets()

    # splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.get_target(n_troll, n_nontroll), test_size=0.1,
                                                        random_state=42, shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD(n_components=10)),
        ("clf", DecisionTreeClassifier())
    ])

    # the combinations to test
    parameter_space = {"vect__ngram_range": [(1, 1), (1, 2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True, False),
                       "clf__criterion": ["gini", "entropy"],
                       "clf": [DecisionTreeClassifier()],
                       "clf__random_state": [42]
                       }
    # definition of the performance metrics
    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll", zero_division=True),
               "npv_score": metrics.make_scorer(metrics.precision_score, pos_label="nontroll", zero_division=True),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "specifity_score": metrics.make_scorer(metrics.recall_score, pos_label="nontroll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    # execute a grid search cross validation with 2 folds
    clf = GridSearchCV(pipe, parameter_space, n_jobs=6, cv=2, scoring=scorers, refit=False, verbose=2)
    clf.fit(X_train, y_train)

    # save the best tuple in order to reuse it for the last comparison
    memory.save(clf.cv_results_, "tree")

    report = HypOptReport("tree", clf.cv_results_)
    report.print()


def print_summary(metr, cargs):
    print("+----------------------------------------------------+")
    print("|              custom hyperparameters                |")
    print("+----------------------------------------------------+")

    t = PrettyTable(header=False)
    t.hrules = ALL
    t.add_row(["technique", "decision tree"])
    t.add_row(["splitting metric", metr])

    t = cargs.get_rows(t)

    print(t)
