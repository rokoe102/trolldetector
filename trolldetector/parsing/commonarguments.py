# wrapper class for the CL arguments all commands are sharing

class CommonArguments:
    def __init__(self, ngram, tfidf, stop, dims, verbose):
        self.ngram = ngram
        self.tfidf = tfidf
        self.stop = stop
        self.dims = dims
        self.verbose = verbose

    def print(self):
        if not self.tfidf:
            print("selected feature weighting: TF")
        else:
            print("selected feature weighting: TF-IDF")
        print("selected n for n-grams: " + str(self.ngram))
        print("filtering english stop words:", end=" ")
        if self.stop:
            print("ON")
        else:
            print("OFF")
        print("selected components for reduction: " + str(self.dims))
