import json
import math

from nltk.corpus import stopwords

from Indexer import Indexer
from Indexer.FileManager import FileManager


class Search:
    def __init__(self):
        pass

    @staticmethod
    def search(query):
        N = FileManager.get_all_file_names('D:\Study Docs\CS\Information Retrieval\Assignment1\InformationRetrieval-master\Corpus').count()
        stop_words = set(stopwords.words('english'))
        tokens = Indexer.Indexer.return_tokens(query, stop_words)
        try:
            index = json.load(open('model'))
        except IOError:
            results = "No index generated yet"
            return results
        token_results = list()
        for token in tokens:
            try:
                data = index[token]
            except KeyError:
                data = None
            if data is not None:
                token_results.append(data)
        docs = {}
        for token_result in token_results:
            idf = N / len(token_result)
            for document in token_result:
                name = document['Name']
                try:
                    weight = docs[name]
                except KeyError:
                    weight = 0
                weight += ((math.log(document['TF']) + 1) * idf)
                docs[name] = weight

        for key in docs:
            weight = docs[key]
            weight /= math.log(open(key).read().count(' '))  # TODO: Count actual words and not spaces
            docs[key] = weight

        results = list()
        for key, value in sorted(docs.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            results.append(key)
        return results

if __name__=="__main__":
    print Search.search('love')

