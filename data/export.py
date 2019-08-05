import pandas as pd
import numpy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

class Export():
    def __init__(self):
        print('init data')

    def read(self,list,path):
        #read the file and output a list
        data_file = open(path,'rt', errors='replace')
        for line in data_file:
            list.append(line.rstrip())
        #output the lsit
        return list

    def print(self,list,n_elements):
        #print last n elements of a list
        print(list[0:n_elements])
        print('\n')

    def create_df(self,list):
        #create and return a pandas dataframe from a list
        df = pd.DataFrame(list)
        return df

    def tokenize_sentence(self,data):
        #obtain all sentences from a list of reviews
        tokenized_sentences=list()
        for sentence in data:
            tokenized_sentences.append(sent_tokenize(sentence))
        return tokenized_sentences

    def tokenize_words(self,data):
        #obtain all words from a list of reviews and return them as a single list
        tokenized_words=list()
        for sentence in data:
            tokenized_words = tokenized_words + word_tokenize(sentence)
        return tokenized_words

    def remove_stopWords(self,data):
        stop_words=set(stopwords.words("english"))
        filtered_words=list()
        for w in data:
            if w not in stop_words:
                filtered_words.append(w)
        return filtered_words

    def stemming(self,data):
        ps = PorterStemmer()
        stemmed_words=list()
        for w in data:
            stemmed_words.append(ps.stem(w))
        return stemmed_words

    def lemmatization(self,data):
        lem = WordNetLemmatizer()
        lemmas=list()
        for w in data:
            lemmas.append(lem.lemmatize(w,'v'))
        return lemmas

    def fdist_words(self,words,name_plot):
        #plot most used words from a word list
        fig=plt.figure()
        fdist = FreqDist(words)
        fdist.plot(30,cumulative=False)
        #plt.show()
        fig.savefig(name_plot)
