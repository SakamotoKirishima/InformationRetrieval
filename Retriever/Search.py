import json
import math
import os
import sys
from pprint import pprint

from nltk.corpus import stopwords

sys.path.append('D:\Documents\Python Scripts\InformationRetrieval\Indexer')

from Indexer.Indexer import Indexer
from Indexer.FileManager import FileManager


# sys.path.insert(0, 'D:\Documents\Python Scripts\InformationRetrieval\Indexer')
# print("test1")


class Search:
    def __init__(self):
        pass

    @staticmethod
    def search(query):
        corpus_dir = '..' + os.sep + 'Corpus'
        N = len(FileManager.get_all_file_names(corpus_dir))
        # exit()
        stop_words = set(stopwords.words('english'))
        tokens = Indexer.return_tokens(query, stop_words)
        # pprint(tokens)
        # exit()
        try:
            fp = open('model.json')
            json_str = fp.read()
            fp.close()
        except IOError:
            results = "No index generated yet"
            return results
        index = json.loads(json_str)
        # pprint(index)
        # exit()
        token_results = list()
        for token in tokens:
            try:
                data = index[token]
                # print(data)
            except KeyError:
                data = None
            if data is not None:
                token_results.append(data)
        docs = {}
        # print(token_results)
        # exit()
        for token_result in token_results:
            idf = N / len(token_result)
            for document in token_result:
                name = document['Name']
                # print(name)
                try:
                    weight = docs[name]
                    # print(weight)
                except KeyError:
                    weight = 0
                # print(document['TF'])
                try:
                    weight += ((math.log(document['TF']) + 1) * idf)
                except ValueError:
                    pass
                docs[name] = weight
        # pprint('DOCS')
        # pprint('______________________________')
        # pprint(docs)
        # pprint('______________________________')
        for key in docs:
            weight = docs[key]
            weight /= math.log(open(key).read().count(' '))  # TODO: Count actual words and not spaces
            docs[key] = weight
        # pprint('DOCS')
        # pprint('______________________________')
        # pprint(docs)
        # pprint('______________________________')
        results = list()
        # for key, value in sorted(docs.items(), key=lambda t: t[1], reverse=True):
        #     # print(key, value)
        #     results.append(key)
        for key, value in sorted(docs.items(), key=lambda x: x[1], reverse=True):
            results.append(key)
        # pprint(sorted_docs)
        return results


if __name__ == "__main__":
    pprint(Search.search('love'))
