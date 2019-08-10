#algorithm  group two: Non deep learning aproaches

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression

from nltk.tokenize import RegexpTokenizer

class Alg2():
    def __init__(self):
        print('init2')

    def get_train_test(self,data):
        #obtain normalized dtm with all tokenized words
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        tf=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
        text_tf= tf.fit_transform(data['Phrase'])

        #generate training and testing sets from raw data and laels (sentiment)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        text_tf, data['Sentiment'], test_size=0.25, random_state=1)

    def train_rawDTM(self,data):
        #it uses the raw Document Term matrix to count the words

        #tokenize data
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        #crate a vectorizer
        cv = CountVectorizer(lowercase=True,ngram_range = (1,1),tokenizer = token.tokenize)
        text_counts= cv.fit_transform(data['Phrase'])
        #generate a training and test set
        X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['Sentiment'], test_size=0.3, random_state=1)
        #fit the model with the train set
        clf = MultinomialNB(alpha=1.53,fit_prior=True).fit(X_train, y_train)
        #evaluate the model (obtain accuracy)
        predicted= clf.predict(X_test)
        print("MultinomialNB Accuracy with raw DTM:",metrics.accuracy_score(y_test, predicted))

    def train_normDTM(self,data):
        #it uses the Term frequency and inverse document frequency to normalize the Document
        #term matrix

        #get train and test
        self.get_train_test(data)
        #fit the model with the train set
        clf = MultinomialNB(alpha=1.53,fit_prior=True).fit(self.X_train, self.y_train)

        #evaluate the model (obtain accuracy)
        predicted= clf.predict(self.X_test)
        print("MultinomialNB Accuracy with normalized DTM:",metrics.accuracy_score(self.y_test, predicted))
        cm = confusion_matrix (self.y_test, predicted)
    def train_SVM(self,data):
        #it uses the Term frequency and inverse document frequency to normalize the Document
        #term matrix

        #get train and test
        self.get_train_test(data)
        #fit the model with the train set
        clf = LinearSVC(random_state=0, tol=1e-5).fit(self.X_train,self.y_train)
        #evaluate the model (obtain accuracy)
        predicted= clf.predict(self.X_test)
        print("SVM Accuracy with normalized DTM:",metrics.accuracy_score(self.y_test, predicted))

    def train_LR(self,data):
        #linear regression algorithm 0.7558 accuracy

        #get train and test
        self.get_train_test(data)
        #summom classifier train and evaluate it
        classifier = LogisticRegression()
        classifier.fit(self.X_train, self.y_train)
        score = classifier.score(self.X_test, self.y_test)

        #print the accuracy with the test set
        print("Accuracy LR:", score)
