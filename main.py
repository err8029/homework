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

    #data preprocessing and sorting
    #----------------------------------------------------------------------

    #data pre processing and saving
    neg=list()
    pos=list()
    neg = obj_data.read(neg,'data/rt-polarity.neg')
    pos = obj_data.read(pos,'data/rt-polarity.pos')

    #check list integrity (print last x elements)
    obj_data.print(neg,1)
    obj_data.print(pos,2)

    #create pandas df from maunal filtered lists
    data = obj_data.create_df(pos,neg)
    print(data)

    #create pandas df from manual filtered lists
    [test,train] = obj_data.create_dfCNN(pos,neg)
    train.info()
    print(train)
    print(test)

    #this part was used for the Statistical analysis of the data
    #for the manual filtering not the classifiers
    #-------------------------------------------------------------

    #tokenize words
    words_neg = obj_data.tokenize_words(neg)
    words_pos = obj_data.tokenize_words(pos)

    #remove stopwords (meaningless words, grammar related, positional...)
    words_neg = obj_data.remove_stopWords(words_neg)
    words_pos = obj_data.remove_stopWords(words_pos)

    #lemmatization, more sophisticated than stemming, returns the correct root word depending on function
    words_neg_l = obj_data.lemmatization(words_neg)
    words_pos_l = obj_data.lemmatization(words_pos)


    #frequency distribution of most used words (just for some curiosity...)
    obj_data.fdist_words(words_neg,'negative_reviews_dis_raw.png')
    obj_data.fdist_words(words_neg_l,'negative_reviews_dis_lemmatization.png')
    obj_data.fdist_words(words_pos_l,'positive_reviews_dis_stemming.png')
    obj_data.fdist_words(words_pos,'positive_reviews_dis_raw.png')




    #algorithms
    #---------------------------------------------------------------------

    #from alg froup 1: MultinomialNB was chosen as it offers the best accuracy
    #and its parameters are easy to tune. Moreover it is a fast classifier that
    #obtains decent results and it is widely used in document analysis
    obj_algorithm2.train_rawDTM(data)
    #from alg group 2: CNN with embedding was chosen. Despite being the most
    #computationally intensive of all, this CNN allows the creation of a dictionary
    #with important words that afterwards will be used to parametrize the embedding
    #laer. Moreover, the hyperparameters are tested and chosen with the random
    #search algorithm, so that the best are chosen
    obj_algorithm1.train_CNN_Emb(test,train,data)



#define main function name
if __name__ == "__main__":
    main()
