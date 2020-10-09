from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
import pandas as pd
import numpy as np

def classify(test, comp,tfidf,cost,verbose):
    if verbose:
        print("--------------------------------------------------------")
        print("classification technique: support-vector machine")
        print("selected cost for misclassification penalization: " + str(cost))
        if tfidf:
            print("selected feature weighting: TF-IDF")
        else:
            print("selected feature weighting: TF")
        print("selected components for reduction: " + str(comp))
        print("training/testing ratio: " + str(1 - test) + "/" + str(test))
        print("--------------------------------------------------------")
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

    count_vec = CountVectorizer()
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

    svd = TruncatedSVD(n_components=comp, random_state=42)
    X_reduced = svd.fit_transform(X_train_counts)

    # splitting into training data and testing data
    if verbose:
        print("splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, target, test_size=test, random_state=42, shuffle=True)

    #svm = SVC(C=cost, kernel="linear")
    svm = LinearSVC(C=cost)

    # training
    if verbose:
        print("training the model")
    svm.fit(X_train, y_train)

    # testing
    if verbose:
        print("making predictions")
    predicted = svm.predict(X_test)

    # report the results
    print("------------------------------------------------------")
    print("                       REPORT                         ")
    print("------------------------------------------------------")

    print(metrics.classification_report(y_test, predicted))