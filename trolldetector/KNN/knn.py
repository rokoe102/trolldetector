from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..memory import memory
from ..parsing import prepare
from ..report.hypoptreport import HypOptReport
from ..report.customreport import CustomReport

from prettytable import PrettyTable, ALL


# use the kNN algorithm with custom hyperparameters
def train_and_test(k, metr,  test, cargs):

    # print a summary of all selected arguments
    print_summary(k,metr,test,cargs)


    if cargs.verbose:
        print("loading datasets")

    tweets, n_troll, n_nontroll = prepare.prepare_datasets()


    count_vec = CountVectorizer(ngram_range=(1,cargs.ngram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and TF-IDF weighting
    if cargs.verbose:
        print("preprocessing")

    X_train_counts = count_vec.fit_transform(tweets)
    if cargs.tfidf:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)


    # dimensionality reduction
    if cargs.verbose:
        print("reducing dimensions")
    svd = TruncatedSVD(n_components=cargs.dims, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)


    # splitting into training data and testing data
    if cargs.verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, prepare.getTarget(n_troll, n_nontroll), test_size=test, random_state=42, shuffle=True)

    knn = KNeighborsClassifier(n_neighbors=k, metric=metr)

    # training
    if cargs.verbose:
        print("training the model")
    knn.fit(X_train, y_train)

    # testing
    if cargs.verbose:
        print("making predictions")
    predicted = knn.predict(X_test)

    # report the results
    report = CustomReport(y_test, predicted)
    report.print()


# performing a hyperparameter optimization for kNN hyperparameters
def optimize(test, verbose):
    print("+---------------------------------------------------------------+")
    print("| hyperparameter optimization for: k-Nearest neighbor algorithm |")
    print("+---------------------------------------------------------------+")
    if verbose:
        print("loading datasets")

    tweets, n_troll, n_nontroll = prepare.prepare_datasets()

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.getTarget(n_troll, n_nontroll), test_size=test, random_state=42,
                                                        shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD(n_components=10)),
        ("clf", KNeighborsClassifier())
    ])

    # the combinations to test
    parameter_space = {"clf": [KNeighborsClassifier()],
                       "vect__ngram_range": [(1,1),(1,2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True,False),
                       "clf__n_neighbors": [5, 15, 25],
                       "clf__metric": ["euclidean", "manhattan"]
                       }

    # definition of the performance metrics
    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll",zero_division=True),
               "npv_score": metrics.make_scorer(metrics.precision_score, pos_label="nontroll",zero_division=True),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "specifity_score": metrics.make_scorer(metrics.recall_score, pos_label="nontroll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    # execute a grid search cross validation with 2 folds
    clf = GridSearchCV(pipe, parameter_space,n_jobs=5,cv=2,scoring=scorers,refit=False,verbose=3)
    clf.fit(X_train, y_train)

    # save the best tuple in order to reuse it for the last comparison
    memory.save(clf.cv_results_, "KNN")

    report = HypOptReport("KNN", clf.cv_results_)
    report.print()

def print_summary(k, metr, test, cargs):
    print("+---------------------------------------------------------------+")
    print("|                   custom hyperparameters                      |")
    print("+---------------------------------------------------------------+")

    t = PrettyTable(header=False)
    t.hrules = ALL
    t.add_row(["technique", "k-nearest neighbor"])
    t.add_row(["k value", k])
    t.add_row(["distance metric", metr])

    t = cargs.get_rows(t)

    t.add_row(["training set", "{} %".format((1 - test) * 100)])
    t.add_row(["test set", "{} %".format(test * 100)])

    print(t)
