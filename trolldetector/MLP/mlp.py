from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from parsing import prepare
from report.hypoptreport import HypOptReport

def trainAndTest(actFunc,test, cargs):

    print("---------------------------------------------------------------")
    print("classification technique: multi-layer perceptron classification")
    cargs.print()
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
    print("---------------------------------------------------------------")
    if cargs.verbose:
        print("loading datasets")

    tweets = prepare.prepare_datasets()

    count_vec = CountVectorizer(ngram_range=(1,cargs.ngram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and weighting
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
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, prepare.getTarget(), test_size=test, random_state=42, shuffle=True)

    mlp = MLPClassifier(random_state=1,activation=actFunc,max_iter=50,verbose=cargs.verbose,early_stopping=True,tol=0.001,n_iter_no_change=5)

    # training
    if cargs.verbose:
        print("training the model")
    mlp.fit(X_train, y_train)

    # testing
    if cargs.verbose:
        print("making predictions")
    predicted = mlp.predict(X_test)

    # report the results
    print("+----------------------------------------------------+")
    print("|                      REPORT                        |")
    print("+----------------------------------------------------+")

    print(metrics.classification_report(y_test, predicted))

def optimize(test, verbose):
    print("+------------------------------------------------------------------------+")
    print("| hyperparameter optimization for: multi-layer perceptron classification |")
    print("+------------------------------------------------------------------------+")
    if verbose:
        print("loading datasets")

    tweets = prepare.prepare_datasets()

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(tweets, prepare.getTarget(), test_size=test, random_state=0)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("reductor", TruncatedSVD()),
        ("clf", MLPClassifier(max_iter=50, tol=0.005, early_stopping=True))
    ])

    parameter_space = {"vect__ngram_range": [(1, 1), (1, 2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True, False),
                       "clf__activation": ["relu","tanh","logistic"],
                       "clf__n_iter_no_change": [5]
                       }

    clf = GridSearchCV(pipe, parameter_space, n_jobs=5, cv=2, verbose=2)
    clf.fit(X_train, y_train)

    report = HypOptReport("MLP", clf.cv_results_)
    report.print()
