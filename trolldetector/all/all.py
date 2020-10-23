from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from parsing import prepare
from report.comparisonreport import ComparisonReport
from memory import memory as mem


def compare(verbose):
    print("+---------------------------------------------------------------+")
    print("|       comparison of all classification techniques             |")
    print("+---------------------------------------------------------------+")

    tweets = prepare.prepare_datasets()

    # splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.getTarget(), test_size=0.1, random_state=42,
                                                        shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD()),
        ("scaling", None),
        ("clf", KNeighborsClassifier())
    ])

    knn_params = mem.load("KNN")
    nb_params = mem.load("NB")
    svm_params = mem.load("SVM")
    tree_params = mem.load("tree")
    mlp_params = mem.load("MLP")

    parameter_space = [knn_params,nb_params,svm_params,tree_params,mlp_params]


    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll"),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    detail = 0
    if verbose:
        detail = 2

    clf = GridSearchCV(pipe, parameter_space, n_jobs=5, cv=2,scoring=scorers,refit=False, verbose=detail)
    clf.fit(X_train, y_train)

    results = clf.cv_results_

    report = ComparisonReport(results)
    report.print()