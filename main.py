#libraries (maybe needed?)
import numpy
import keras
import sklearn


#my own packages per each method
from alg1.main import Alg1
from alg2.main import Alg2
from data.export import Export

def main():
    print('starting...')

    #init objs per each method
    #---------------------------------------------------------------------

    obj_data=Export()
    obj_algorithm1=Alg1()
    obj_algorithm2=Alg2()

    #exec methods here :)
    #----------------------------------------------------------------------

    #data pre processing and saving
    try:
        neg=list()
        pos=list()
        neg = obj_data.read(neg,'data/rt-polarity.neg')
        pos = obj_data.read(pos,'data/rt-polarity.pos')

        #check list integrity (print last x elements)
        obj_data.print(neg,1)
        obj_data.print(pos,2)

        #create pandas df from filtered lists
        df_neg = obj_data.create_df(neg)
        df_pos = obj_data.create_df(pos)

        #tokenize words
        words_neg = obj_data.tokenize_words(neg)
        words_pos = obj_data.tokenize_words(pos)

        #remove stopwords (meaningless words, grammar related, positional...)
        words_neg = obj_data.remove_stopWords(words_neg)
        words_pos = obj_data.remove_stopWords(words_pos)

        #stemming (obtain the root of the words)
        words_neg_s = obj_data.stemming(words_neg)
        words_pos_s = obj_data.stemming(words_pos)

        #lemmatization, more sophisticated than stemming, returns the correct root word depending on function
        words_neg_l = obj_data.lemmatization(words_neg)
        words_pos_l = obj_data.lemmatization(words_pos)

        #frequency distribution of most used words (just for some curiosity...)
        obj_data.fdist_words(words_neg_s,'negative_reviews_dis_stemming.png')
        obj_data.fdist_words(words_neg_l,'negative_reviews_dis_lemmatization.png')
        obj_data.fdist_words(words_pos_s,'positive_reviews_dis_stemming.png')
        obj_data.fdist_words(words_pos_l,'positive_reviews_dis_lemmatization.png')

    except Exception as error:
        print('sth went wrong in data grabbing and processing')
        print(error)


    obj_algorithm2.exec()
    obj_algorithm1.exec()


#define main function name
if __name__ == "__main__":
    main()
