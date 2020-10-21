from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from parsing import prepare
from report.hypoptreport import HypOptReport


# use the KNN method with custom hyperparameters
def trainAndTest(k, metr,  test, cargs):

    print("+--------------------------------------------------------+")
    print("classification technique: k-nearest neighbor algorithm")
    print("selected k value: " + str(k))
    print("selected metric for distance measurement: " + metr)
    cargs.print()
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
    print("+--------------------------------------------------------+")
    if cargs.verbose:
        print("loading datasets")

    tweets = prepare.prepare_datasets()


    count_vec = CountVectorizer(ngram_range=(1,cargs.ngram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and TF-IDF weighting
    if cargs.verbose:
        print("preprocessing")

    X_train_counts = count_vec.fit_transform(tweets)
    if cargs.tfidf == True:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)


    # dimensionality reduction
    if cargs.verbose:
        print("reducing dimensions")
    svd = TruncatedSVD(n_components=cargs.dims, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)


    # splitting into training data and testing data
    if cargs.verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, prepare.getTarget(), test_size=test, random_state=42, shuffle=True)

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
    print("+----------------------------------------------------+")
    print("|                      REPORT                        |")
    print("+----------------------------------------------------+")

    print(metrics.classification_report(y_test, predicted))


# hyperparameter optimization for KNN hyperparameters

def optimize(test, verbose):
    print("+---------------------------------------------------------------+")
    print("| hyperparameter optimization for: k-Nearest neighbor algorithm |")
    print("+---------------------------------------------------------------+")
    if verbose:
        print("loading datasets")

    tweets = prepare.prepare_datasets()

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.getTarget(), test_size=test, random_state=42,
                                                        shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD()),
        ("clf", KNeighborsClassifier())
    ])

    parameter_space = {"vect__ngram_range": [(1,1),(1,2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True,False),
                       "clf__n_neighbors": [5, 13],
                       "clf__metric": ["euclidean", "manhattan", "chebyshev"],
    }

    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll"),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score)
              }

    # execute a grid search: testing every combination of hyperparameters

    clf = GridSearchCV(pipe, parameter_space,n_jobs=6,cv=2,scoring=scorers,refit=False,verbose=3)
    clf.fit(X_train, y_train)

    report = HypOptReport("KNN", clf.cv_results_)
    report.print()