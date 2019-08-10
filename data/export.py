import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter

class Export():
    def __init__(self):
        print('init data')

    def replaceMultiple(self,mainString, toBeReplaces, newString):
        # Iterate over the strings to be replaced
        for elem in toBeReplaces :
            # Check if string is in the main string
            if elem in mainString :
                # Replace the string
                mainString = mainString.replace(" "+elem+" ", newString)
            if mainString.startswith(elem+" "):
                # Replace the string at the start
                mainString = mainString[len(elem+" "):]
            if mainString.endswith(" "+elem):
                # Replace the string at the end
                mainString = mainString[:len(" "+elem)]

        return  mainString
    def read(self,list,path):
        #read the file (forcing utf8) and output a list
        lemmer=WordNetLemmatizer()

        #count the words ,save those that have a frequency of 15 or less
        #so that the algorithm doesn't get distracted by irrellevant non-recurrent words
        data_file = open(path,'rt', encoding='utf8', errors='replace')
        cn = Counter(word for l in data_file for word in l.split())
        words=dict((word,v )for word,v in cn.items() if v < 6 )
        words_list=words.keys()
        data_file.close()

        #read the stopwords file and create a list with them
        stopwords_f = open('stopwords.txt','rt', encoding='utf8', errors='replace')
        stopwords=[]
        for line in stopwords_f:
            stopwords.append(str(line.strip()))
        stopwords_f.close()

        data_file = open(path,'rt', encoding='utf8', errors='replace')
        for line in data_file:
            #manually check for stopwords
            new_line=''

            #manually eliminate puntuation and grammar stuff, according to stastics
            #of the data, least repeated and most repeated words (irrellevant ones)
            line = str(line).replace(","," ")
            line = str(line).replace(' " ',' ')
            line = str(line).replace(".","")
            line = str(line).replace(" ?","")
            line = str(line).replace(" : "," ")
            line = str(line).replace(" ; "," ")
            line = str(line).replace(" ( "," ")
            line = str(line).replace(" ) "," ")
            line = str(line).replace(". . ."," ")
            line = str(line).replace(" -- "," ")


            #remove some stopwords manually as well as digits from the strings
            line = self.replaceMultiple(str(line), words_list,' ')
            #remove least used words (words that have a frequency of 3 and less)
            line = self.replaceMultiple(str(line), stopwords,' ')

            #lematize the final words in the line
            for word in line.split(' '):
                new_line = new_line + ' ' + str(lemmer.lemmatize(word))
            list.append(line.rstrip())

        data_file.close()
        #output the lsit
        return list

    def print(self,list,n_elements):
        #print last n elements of a list
        print(list[0:n_elements])
        print('\n')

    def create_df(self,list_pos,list_neg):
        #create and return a pandas dataframe from a list with all phrases and its evaluation

        list_total=list_pos+list_neg
        sentiments=[1]*len(list_pos)+[0]*len(list_neg)

        data=pd.DataFrame({'Phrase':list_total,'Sentiment':sentiments})
        return data

    def create_dfCNN(self,list_pos,list_neg):
        #return two df, one for testing and the other for training, designed for the CNN
        #with embedding

        neg_train=list_neg[len(list_neg)//3:]
        neg_test=list_neg[:len(list_neg)//3]

        pos_train=list_pos[len(list_pos)//3:]
        pos_test=list_pos[:len(list_pos)//3]

        test=neg_test+pos_test
        train=neg_train+pos_train

        sentiments_test=[0]*len(neg_test)+[1]*len(pos_test)
        sentiments_train=[0]*len(neg_train)+[1]*len(pos_train)

        test=pd.DataFrame({'Phrase':test,'Sentiment':sentiments_test})
        train=pd.DataFrame({'Phrase':train,'Sentiment':sentiments_train})

        return [test,train]
    def tokenize_sentences(self,data):
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
        #remove stopwords automatically in every line

        stop_words=set(stopwords.words("english"))
        filtered_words=list()
        for w in data:
            if w not in stop_words:
                filtered_words.append(w)
        return filtered_words

    def lemmatization(self,data):
        #obtain the lemmas for every single word in every single line

        lem = WordNetLemmatizer()
        lemmas=list()
        for w in data:
            lemmas.append(lem.lemmatize(w,'v'))
        return lemmas

    def fdist_words(self,words,name_plot):
        #plot most used words from a word list

        fig=plt.figure()
        fdist = FreqDist(words)
        fdist.plot(40,cumulative=False)
        fig.savefig(name_plot)
