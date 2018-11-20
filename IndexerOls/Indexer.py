import json
import math
from pprint import pprint

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from FileManager import FileManager


def intersect(l1, l2):
    return list(value for value in l1 if value in l2)


class Indexer:
    def __init__(self):
        pass

    @staticmethod
    def return_tokens(file_contents, stopw):
        file_contents = file_contents.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = set(tokenizer.tokenize(file_contents))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        tokens_new = list()
        for word in tokens:
            word = stemmer.stem(word)
            word = lemmatizer.lemmatize(word)
            tokens_new.append(word)
        tokens_new = Indexer.remove_stop_words(tokens_new, stopw)
        tokens_new = set(tokens_new)
        return tokens_new

    @staticmethod
    def return_tf(fo, w):
        s = fo.read()
        tf_t_d = s.count(w)
        return tf_t_d  # 1 + math.log10(tf_t_d)

    @staticmethod
    def return_idf(document_list, term):
        no_of_documents = len(document_list)
        df_t = 0
        for document in document_list:
            fo = open(document)
            s = fo.read()
            if s.find(term) != -1:
                df_t += 1
        return math.log10(float(no_of_documents) / float(df_t))

    @staticmethod
    def create_matrix(directory, index):
        stop_words = set(stopwords.words('english'))
        file_name_list = FileManager.get_all_file_names(directory)  # I changed it from a generator since we
        # need the number of documents

        for file_name in file_name_list:
            # make sure it runs on one first, then we'll index the entire corpus
            file_object = open(file_name)
            file_content = file_object.read()
            tokens = Indexer.return_tokens(file_content, stop_words)
            for token in tokens:
                try:
                    current_data = index[token]
                except KeyError:
                    current_data = list()
                to_add = {'Name': file_name, 'TF': Indexer.return_tf(file_object, token)}
                current_data.append(to_add)
                index[token] = current_data
            # exit()
            # return index
        return index

    @staticmethod
    def remove_stop_words(l, words_to_delete):
        stopwords_in_dict = intersect(l, words_to_delete)
        for word in stopwords_in_dict:
            l.remove(word)
        return l


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# The above 3 download the requirements for any nltk methods I have used to
# function properly. They need to be run the first time, after which you can comment them out
folder = '..\Corpus'  # Keep Corpus and Scraper in the same directory
# pprint(FileManager.get_all_file_names(folder))
# exit()
# fo= open('model.json', 'w')
# fo.write('')
# exit()
# try:
#    ind = json.load(open('model.json'))
# except IOError:
#    ind = {}
ind = {}
ind = Indexer.create_matrix(folder, ind)
fo = open('model.json', 'w')
pprint(ind)
json.dump(ind, fo)
