from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from parsing import prepare
def trainAndTest(test,metr,cargs):

    print("--------------------------------------------------------")
    print("classification technique: decision tree classification")
    print("selected metric for finding the best split: " + metr)
    cargs.print()
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
    print("--------------------------------------------------------")
    if cargs.verbose:
        print("loading datasets")

    # load and label datasets
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

    treeClf = DecisionTreeClassifier(criterion=metr)

    # training
    if cargs.verbose:
        print("training the model")
    treeClf.fit(X_train, y_train)

    # testing
    if cargs.verbose:
        print("making predictions")
    predicted = treeClf.predict(X_test)

    # report the results
    print("------------------------------------------------------")
    print("                       REPORT                         ")
    print("------------------------------------------------------")

    print(metrics.classification_report(y_test, predicted))

def optimize(test, verbose):
    print("-------------------------------------------------------------")
    print("hyperparameter optimization for: decision tree classification")
    print("-------------------------------------------------------------")
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
        ("clf", DecisionTreeClassifier())
    ])

    parameter_space = {"vect__ngram_range": [(1, 1), (1, 2)],
         "vect__stop_words": [None, "english"],
         "tfidf__use_idf": (True, False),
         "reductor": [TruncatedSVD()],
         "reductor__n_components": [10],
         "clf__criterion": ["entropy"]
         }

    grSearch = GridSearchCV(pipe, parameter_space, n_jobs=5, cv=2, verbose=2)
    grSearch.fit(X_train, y_train)

    print("Best score: %0.3f" % grSearch.best_score_)
    print("Best parameters set:")
    print(grSearch.best_params_)
