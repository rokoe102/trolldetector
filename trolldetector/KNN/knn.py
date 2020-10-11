from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# use the KNN method with own selected hyperparameters

def trainAndTest(k, metr, tf,gram, test, components, verbose):

    if verbose:
        print("------------------------------------------------------")
        print("classification technique: k-nearest neighbor algorithm")
        print("selected k value: " + str(k))
        print("selected metric for distance measurement: " + metr)
        if tf:
            print("selected feature weighting: TF")
        else:
            print("selected feature weighting: TF-IDF")
        print("selected n for n-grams: " + str(gram))
        print("selected components for reduction: " + str(components))
        print("training/testing ratio: " + str(1 - test) + "/" + str(test))
        print("------------------------------------------------------")
        print("loading datasets")

    troll = pd.read_csv("datasets/troll_top10.csv", encoding="utf-8", low_memory=False)
    nontroll = pd.read_csv("datasets/nontroll_top10.csv", encoding="utf-8",low_memory=False)

    # merge and label
    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()
    tweets.extend(nontroll)
    target = np.full(303036, "troll").tolist()
    t = np.full(324873, "nontroll").tolist()
    target.extend(t)


    count_vec = CountVectorizer(ngram_range=(1,gram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and TF-IDF weighting
    if verbose:
        print("preprocessing")

    X_train_counts = count_vec.fit_transform(tweets)
    if tf == False:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)


    # dimensionality reduction
    if verbose:
        print("reducing dimensions")
    svd = TruncatedSVD(n_components=components, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)


    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, target, test_size=test, random_state=42, shuffle=True)

    knn = KNeighborsClassifier(n_neighbors=k, metric=metr)

    # training
    if verbose:
        print("training the model")
    knn.fit(X_train, y_train)

    # testing
    if verbose:
        print("making predictions")
    predicted = knn.predict(X_test)

    # report the results
    print("------------------------------------------------------")
    print("                       REPORT                         ")
    print("------------------------------------------------------")

    print(metrics.classification_report(y_test, predicted))

def optimize(tf, test, components, verbose):
    print("-------------------------------------------------------------")
    print("hyperparameter optimization for: k-Nearest neighbor algorithm")
    print("-------------------------------------------------------------")
    if verbose:
        print("loading datasets")

    troll = pd.read_csv("datasets/troll_top10.csv", encoding="utf-8", low_memory=False)
    nontroll = pd.read_csv("datasets/nontroll_top10.csv", encoding="utf-8", low_memory=False)

    # merge and label
    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()
    tweets.extend(nontroll)
    target = np.full(303036, "troll").tolist()
    t = np.full(324873, "nontroll").tolist()
    target.extend(t)

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=test, random_state=42,
                                                        shuffle=True)

    pipe = Pipeline(steps=[
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        #("scale", StandardScaler()),
        ("reductor", TruncatedSVD()),
        ("clf", KNeighborsClassifier())
    ])

    parameter_space = {"vect__ngram_range": [(1,1),(1,2)],
                       "tfidf__use_idf": (True,False),
                       "clf__n_neighbors": [5],
                       "clf__metric": ["euclidean", "manhattan", "chebyshev"],

    }

    grSearch = GridSearchCV(pipe, parameter_space,n_jobs=4,verbose=2)
    grSearch.fit(X_train, y_train)

    print("Best score: %0.3f" % grSearch.best_score_)
    print("Best parameters set:")
    best_parameters = grSearch.best_estimator_.get_params()
    for param_name in sorted(parameter_space.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

