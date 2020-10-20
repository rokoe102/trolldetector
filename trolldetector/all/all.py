from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from parsing import prepare

def compare():
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
        ("clf", KNeighborsClassifier())
    ])

    parameter_space = [
                      {"clf": [KNeighborsClassifier(n_neighbors=10, metric="euclidean")],
                       "vect__ngram_range": [(1,1)],
                       "vect__stop_words": ["english"],
                       "tfidf__use_idf": [False]
                      },
                      {"clf": [GaussianNB()],
                       "tfidf__use_idf": [False],
                       "vect__ngram_range": [(1,2)],
                       "vect__stop_words": ["english"]
                      },
                      {"clf": [LinearSVC(C=1)],
                       "tfidf__use_idf": [False],
                       "vect__ngram_range": [(1,2)],
                       "vect__stop_words": [None]
                      },
                      {"clf": [DecisionTreeClassifier(criterion="entropy")],
                       "tfidf__use_idf": [False],
                       "vect__ngram_range": [(1,1)],
                       "vect__stop_words": ["english"]
                      },
                      {"clf": [MLPClassifier(activation="relu", early_stopping=True,tol=0.005, n_iter_no_change=5)],
                       "tfidf__use_idf": [False],
                       "vect__ngram_range": [(1,2)],
                       "vect__stop_words": [None]
                      }
    ]

    #clf = GridSearchCV(pipe, parameter_space, n_jobs=7, cv=4, verbose=2)
    clf = GridSearchCV(pipe, parameter_space, n_jobs=7, cv=3, verbose=2)
    clf.fit(X_train, y_train)


    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("|                      REPORT                        |")
    print("|++++++++++++++++++++++++++++++++++++++++++++++++++++|")
    print("|             ranking of classificators              |")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    results = clf.cv_results_

    rank = 1
    zipped = sorted(zip(results["mean_test_accuracy_score"], results["std_test_accuracy_score"], results["params"]), key=lambda t: t[0],
                    reverse=True)
    for mean, std, params in zipped:
        print("[" + str(rank) + "]", end=" ")
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params["clf"]))
        rank += 1

    ### als nächstes: andere Scoring-Metriken für alle Reports einfügen