# wrapper class for the CL arguments all commands are sharing

class CommonArguments:
    def __init__(self, ngram, tfidf, stop, dims, verbose):
        self.ngram = ngram
        self.tfidf = tfidf
        self.stop = stop
        self.dims = dims
        self.verbose = verbose

    def get_rows(self, table):
        if self.tfidf:
            table.add_row(["feature weighting", "TF-IDF"])
        else:
            table.add_row(["feature weighting", "TF"])

        table.add_row(["ngram-range", "(1," + str(self.ngram) + ")"])
        if self.stop:
            table.add_row(["stopwords", "filtered"])
        else:
            table.add_row(["stopwords", "not filtered"])
        table.add_row(["dimensions", self.dims])

        return table

