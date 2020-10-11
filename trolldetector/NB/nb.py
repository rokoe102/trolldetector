from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.decomposition import NMF

import pandas as pd
import numpy as np

def trainAndTest(test, comp,tfidf,gram,dist, verbose):

    if verbose:
        print("------------------------------------------------------")
        print("classification technique: Naive Bayes classifyer")
        print("presumed distribution: " + dist)
        if tfidf:
            print("selected feature weighting: TF-IDF")
        else:
            print("selected feature weighting: TF")
        print("selected n for n-grams: " + str(gram))
        print("selected components for reduction: " + str(comp))
        print("training/testing ratio: " + str(1 - test) + "/" + str(test))
        print("------------------------------------------------------")
        print("loading datasets")

    troll = pd.read_csv("datasets/troll_top10.csv", encoding="utf-8", low_memory=False)
    nontroll = pd.read_csv("datasets/nontroll_top10.csv", encoding="utf-8", low_memory=False)

    # merge both
    tweets = troll['content'].tolist()
    nontroll = nontroll['Text'].tolist()
    tweets.extend(nontroll)

    # create labels
    target = np.full(303036, "troll").tolist()
    t = np.full(324873, "nontroll").tolist()
    target.extend(t)

    count_vec = CountVectorizer(ngram_range=(1,gram))
    tfidf_transformer = TfidfTransformer()

    # vectorizing and weighting
    if verbose:
        print("preprocessing")

    X_train_counts = count_vec.fit_transform(tweets)
    if tfidf:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)

    # dimensionality reduction
    if verbose:
        print("reducing dimensions")

    #svd = TruncatedSVD(n_components=comp, random_state=42)
    #X_reduced = svd.fit_transform(X_train_counts)

    nmf = NMF(n_components=comp)
    X_reduced = nmf.fit_transform(X_train_counts)


    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, target, test_size=test, random_state=42,
                                                        shuffle=True)

    if dist == "gaussian":
        gnb = GaussianNB()

        # training
        if verbose:
            print("training the model")
        gnb.fit(X_train, y_train)

        # testing
        if verbose:
            print("making predictions")
        predicted = gnb.predict(X_test)

        # report the results
        print("------------------------------------------------------")
        print("                       REPORT                         ")
        print("------------------------------------------------------")

        print(metrics.classification_report(y_test, predicted))

    elif dist == "multinomial":
        mnb = MultinomialNB()
        # training
        if verbose:
            print("training the model")
        mnb.fit(X_train, y_train)

        # testing
        if verbose:
            print("making predictions")
        predicted = mnb.predict(X_test)

        # report the results
        print("------------------------------------------------------")
        print("                       REPORT                         ")
        print("------------------------------------------------------")

        print(metrics.classification_report(y_test, predicted))

def optimize(tfidf, test, components, verbose):
    print("-------------------------------------------------------------")
    print("hyperparameter optimization for: k-Nearest neighbor algorithm")
    if tfidf:
        print("selected feature weighting: TF-IDF")
    else:
        print("selected feature weighting: TF")
    print("selected components for reduction: " + str(components))
    print("training/testing ratio: " + str(1 - test) + "/" + str(test))
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

    count_vec = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    # vectorizing and TF-IDF weighting
    X_train_counts = count_vec.fit_transform(tweets)
    if tfidf == True:
        X_train_counts = tfidf_transformer.fit_transform(X_train_counts)

    # dimensionality reduction
    svd = TruncatedSVD(n_components=components, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, target, test_size=test, random_state=42,
                                                        shuffle=True)

    nb = GaussianNB()

    parameter_space = {"n_neighbors": [1, 2, 5, 7, 9],
                       "metric": ["euclidean", "manhattan", "chebyshev"],
                       "weights": ["uniform", "distance"]
                       }

    clf = GridSearchCV(knn, parameter_space, n_jobs=-1, cv=3, verbose=verbose)
    clf.fit(X_train, y_train)

    # Best parameter set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
