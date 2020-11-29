from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from memory import memory
from parsing import prepare
from report.hypoptreport import HypOptReport


# use the naive Bayes classification with custom hyperparameters
def train_and_test(test,dist,cargs):

    print("+----------------------------------------------------+")
    print("classification technique: Naive Bayes classifyer")
    print("presumed distribution: " + dist)
    cargs.print()
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
    print("+----------------------------------------------------+")
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

    svd = TruncatedSVD(n_components=cargs.dims)
    X_reduced = svd.fit_transform(X_train_counts)

    
    if dist in ["multinomial", "complement", "categorical"]:
        if cargs.verbose:
            print("scaling feature vectors")
        minmax = MinMaxScaler(feature_range=(0,1))
        X_reduced = minmax.fit_transform(X_reduced)

    # splitting into training data and testing data
    if cargs.verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, prepare.getTarget(), test_size=test, random_state=42,
                                                        shuffle=True)
    predicted = []

    if dist == "gaussian":
        gnb = GaussianNB()

        # training
        if cargs.verbose:
            print("training the model")
        gnb.fit(X_train, y_train)

        # testing
        if cargs.verbose:
            print("making predictions")
        predicted = gnb.predict(X_test)

    elif dist == "bernoulli":
        bnb = BernoulliNB()
        # training
        if cargs.verbose:
            print("training the model")
        bnb.fit(X_train, y_train)

        # testing
        if cargs.verbose:
            print("making predictions")
        predicted = bnb.predict(X_test)


    elif dist == "multinomial":
        mnb = MultinomialNB()
        # training
        if cargs.verbose:
            print("training the model")
        mnb.fit(X_train, y_train)

        # testing
        if cargs.verbose:
            print("making predictions")
        predicted = mnb.predict(X_test)


    elif dist == "complement":
        com = ComplementNB()
        # training
        if cargs.verbose:
            print("training the model")
        com.fit(X_train, y_train)

        # testing
        if cargs.verbose:
            print("making predictions")
        predicted = com.predict(X_test)


    # report the results
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("|                      REPORT                        |")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # print the entries of the confusion matrix

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()
    print("true negatives: " + str(tn))
    print("false negatives: " + str(fn))
    print("true positives: " + str(tp))
    print("false positives: " + str(fp))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # print a classification report with the results for all performance metrics

    print(metrics.classification_report(y_test, predicted,zero_division=1))


# performing a hyperparameter optimization for the naive Bayes classification
def optimize(test, verbose):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("| hyperparameter optimization for: Naive Bayes classification |")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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
        ("scaling", MinMaxScaler()),
        ("clf", GaussianNB())
    ])

    # the combinations to test
    parameter_space = [
                      {"vect__ngram_range": [(1, 1), (1, 2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True, False),
                       "scaling": [None],
                       "clf": [GaussianNB()],
                      },
                      {
                       "vect__ngram_range": [(1,1),(1,2)],
                       "vect__stop_words": [None, "english"],
                       "tfidf__use_idf": (True, False),
                       "scaling": [MinMaxScaler()],
                       "clf": [MultinomialNB(),ComplementNB()],
                      }
    ]

    # definition of the performance metrics
    scorers = {"precision_score": metrics.make_scorer(metrics.precision_score, pos_label="troll",zero_division=True),
               "npv_score": metrics.make_scorer(metrics.precision_score, pos_label="nontroll",zero_division=True),
               "recall_score": metrics.make_scorer(metrics.recall_score, pos_label="troll"),
               "specifity_score": metrics.make_scorer(metrics.recall_score, pos_label="nontroll"),
               "accuracy_score": metrics.make_scorer(metrics.accuracy_score),
               "f1_score": metrics.make_scorer(metrics.f1_score, pos_label="troll")
               }

    # execute a grid search cross validation with 2 folds

    clf = GridSearchCV(pipe, parameter_space, n_jobs=5,cv=2,scoring=scorers,refit=False, verbose=2)
    clf.fit(X_train, y_train)


    # save the best tuple in order to reuse it for the last comparison
    memory.save(clf.cv_results_, "NB")



    report = HypOptReport("NB", clf.cv_results_)
    report.print()
