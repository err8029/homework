#algorithm  group one: Deep learning aproaches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers
import keras

from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt
import numpy as np

class Alg1():
    def __init__(self):
        print('')

    def get_train_test(self,data):
        #obtain normalized dtm with all tokenized words
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        tf=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize, norm='l2', binary=True, max_df=0.3)
        text_tf= tf.fit_transform(data['Phrase'])

        #generate training and testing sets from raw data and laels (sentiment)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        text_tf, data['Sentiment'], test_size=0.25, random_state=1)

    def create_model(self,num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
        #creates a CNN model with the specified params, returns the model oect itself
        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        model.summary()
        return model

    def plot_history(self,history):
        #plot evolution of training of CNN (all epochs)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def train_CNN(self,data):
        #very basic NN (mpt a lot of layers)
        plt.style.use('ggplot')

        #get train and test
        self.get_train_test(data)
        #generate compile and print the NN model with ust a dense layer
        input_dim = X_train.shape[1]  # Number of features
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        #train the model with the previously generated set, returns history (evolution of the Training
        #during all the epochs)
        history = model.fit(self.X_train, self.y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)

        #print all the rellevant results (testing accuracy, training accuracy)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        #show the evolution of the praining in a asic plot
        plt.style.use('ggplot')
        self.plot_history(history)

    def train_CNN_Emb(self,data_test,data_train,data):
        #we represent words as dense vectors (emeddings), this emeddings map the words
        #according to their semmantic information

        #we use the tokenizer over the raw data (splited manually)
        #the output will be a dictionary conatining the most common words till num_of_words
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(data_train['Phrase'])

        #convert the tokenized words into integrers, usale for the network
        X_train = tokenizer.texts_to_sequences(data_train['Phrase'])
        X_test = tokenizer.texts_to_sequences(data_test['Phrase'])
        #lenght of the mentioned dictionary is stored
        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 100
        #to ensure that both sequences are the same lenght we add zeros of a max lenght
        #of 100 ints
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        #some basic data for the CNN training and layers
        embedding_dim = 50
        epochs=30

        # Parameter grid for grid search (using random search alg)
        param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[50],
                      maxlen=[maxlen])

        #create a model oect with some more specs of the model training
        #and model creation function
        model = KerasClassifier(build_fn=self.create_model,
                            epochs=epochs, batch_size=1000,
                            verbose=False)

        #randomized search using the model object, the parameters dict, each set of parameters
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)

        #grab the label sets from the raw data
        y_train = data_train['Sentiment']
        y_test = data_test['Sentiment']

        #test every single model with every single param
        grid_result = grid.fit(X_train, y_train)
        test_accuracy = grid.score(X_test, y_test)

        #generate an string with all the best results and params and print it
        s = ('Best Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
             grid_result.best_score_,
             grid_result.best_params_,
             test_accuracy)
        print(output_string)

        #create a new model object with the best params
        model = self.create_model(grid_result.best_params_['num_filters'],
                                grid_result.best_params_['kernel_size'],
                                grid_result.best_params_['vocab_size'],
                                grid_result.best_params_['embedding_dim'],
                                grid_result.best_params_['maxlen'])

        #train it now with more epochs
        history = model.fit(X_train, y_train,
                    epochs=30,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=1000)

        #print relevant information and plot the evolution of the training
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        self.plot_history(history)
