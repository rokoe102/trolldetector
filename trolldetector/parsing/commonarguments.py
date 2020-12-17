# wrapper class for the CL arguments all commands are sharing

class CommonArguments:
    def __init__(self, ngram, tfidf, stop, dims, test_set):
        self.ngram = ngram
        self.tfidf = tfidf
        self.stop = stop
        self.dims = dims
        self.test_set = test_set

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

        table.add_row(["training set", "{} %".format((1 - self.test_set) * 100)])
        table.add_row(["test set", "{} %".format(self.test_set * 100)])

        return table
