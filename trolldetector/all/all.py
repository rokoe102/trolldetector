from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ..memory import memory as mem
from ..parsing import prepare
from ..report.comparisonreport import ComparisonReport


# performing a grid search over all 5 classification techniques
# in order to compare their performance

def compare():
    print("+---------------------------------------------------------------+")
    print("|       comparison of all classification techniques             |")
    print("+---------------------------------------------------------------+")

    tweets, n_troll, n_nontroll = prepare.prepare_datasets()

    # splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.get_target(n_troll, n_nontroll), test_size=0.1,
                                                        random_state=42,
                                                        shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD(n_components=10)),
        ("scaling", None),
        ("clf", KNeighborsClassifier())
    ])

    # loading the best tuples from the hyperparameter optimizations
    knn_params = mem.load("KNN")
    nb_params = mem.load("NB")
    svm_params = mem.load("SVM")
    tree_params = mem.load("tree")
    mlp_params = mem.load("MLP")

    parameter_space = [knn_params, nb_params, svm_params, tree_params, mlp_params]

    # definition of the performance metrics
    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll"),
               "npv_score": metrics.make_scorer(metrics.precision_score, pos_label="nontroll"),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "specifity_score": metrics.make_scorer(metrics.recall_score, pos_label="nontroll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    # execute a grid search cross validation with 2 folds
    clf = GridSearchCV(pipe, parameter_space, n_jobs=5, cv=2, scoring=scorers, refit=False, verbose=2)
    clf.fit(X_train, y_train)

    results = clf.cv_results_

    report = ComparisonReport(results)
    report.print()
