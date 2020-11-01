from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from parsing import prepare
from report.hypoptreport import HypOptReport
from memory import memory

def trainAndTest(actFunc,iter, tol,test, cargs):

    print("+-------------------------------------------------------------+")
    print("classification technique: multi-layer perceptron classification")
    cargs.print()
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
    print("+-------------------------------------------------------------+")
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

    mlp = MLPClassifier(random_state=42,activation=actFunc,max_iter=50,verbose=cargs.verbose,early_stopping=True,tol=tol,n_iter_no_change=iter)

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

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()
    print("true negatives: " + str(tn))
    print("false negatives: " + str(fn))
    print("true positives: " + str(tp))
    print("false positives: " + str(fp))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

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
        ("reductor", TruncatedSVD(n_components=10)),
        ("clf", MLPClassifier())
    ])

    parameter_space = {"vect__ngram_range": [(1, 1), (1, 2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True, False),
                       "clf__activation": ["relu","tanh","logistic"],
                       "clf__n_iter_no_change": [5],
                       "clf__max_iter": [50],
                       "clf__tol": [0.001],
                       "clf__early_stopping": [True],
                       "clf": [MLPClassifier()],
                       "clf__random_state": [42]
                       }

    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll",zero_division=True),
               "npv_score": metrics.make_scorer(metrics.precision_score, pos_label="nontroll",zero_division=True),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "specifity_score": metrics.make_scorer(metrics.recall_score, pos_label="nontroll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    clf = GridSearchCV(pipe, parameter_space, n_jobs=5, cv=2,scoring=scorers,refit=False, verbose=2)
    clf.fit(X_train, y_train)

    memory.save(clf.cv_results_, "MLP")

    report = HypOptReport("MLP", clf.cv_results_)
    report.print()
