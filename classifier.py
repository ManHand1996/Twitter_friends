import json
import os
from twitters import get_tweepyAPI
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import BernoulliNB # 二值特征分类器
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
# output_filename = os.path.join('./Data','python_tweets.json')



class NLTKBOW(TransformerMixin):
    """
        NLTK a data transformer for document
        rewrite TransformerMixin fit() and transform() functions
    """
    def fit(self,X,y=None,**fit_params):
        return self
    
    def transform(self,X):
        return [{ word:True for word in word_tokenize( document ) }
                for document in X]

class TweetClassifier():

    def __init__(self,proxy=''):
        self.t = get_tweepyAPI(proxy) 
    
    def __get_tweets_samples(self,theme,tweets_filename,nums=100):
        """
            get new tweets from twitter API by theme
        """
        with open (tweets_filename,'a') as output:
            search_results = self.t.search(q=theme,count=nums)
            tweets = []
            for tweet in search_results:
                if 'text' in tweet._json:
                    tweets.append(tweet._json)
                    output.write(json.dumps(tweet._json))
                    output.write("\n\n")
            return tweets

    

    def get_labels(self,theme,tweets_filename,labels_filename,nums=100):
        """
            input 1 or 0 to decide current text is related or not to theme
            theme:twitter API search topic
            tweets_filename: your tweets file str path
            labels_filename: your labels file str path
        """
        tweets = self.__get_tweets_samples(theme=theme,tweets_filename=tweets_filename,nums=nums)
        labels = []
        print("input 1 or 0 to current text is related to theme")
        for i,tweet in enumerate(tweets):
            print(tweet['text'])
            r = input('{0}.input {1} or {2}--->:'.format(i+1,0,1))
            while(not (len(r) == 1 and r.isdigit())):
                r = input('{0}.retry input {1} or {2}--->:'.format(i+1,0,1))
            labels.append(int(r))
        with open(labels_filename,'a') as out:
            json.dump(labels,out)

    
    def load_laebls(self,labels_filename):
        """
            return labels
            labels_filename: your labels file str path
        """
        labels = []
        with open(labels_filename,'r') as inf:
            labels = json.load(inf)
        print("Loaded {0} labels.".format(len(labels)))
        return labels

    
    def load_tweets(self,tweets_filename):
        """
            return tweets
            tweets_filename: your tweets file str path
        """
        with open(tweets_filename,'r') as info:
            tweets = []
            for tweet in info:
                if len(tweet.strip()) == 0:
                    continue
                tweets.append(json.loads(tweet))
            # strip the same tweets
            # tweets = list(set(tweets))
            print("Loaded {0} tweets.".format(len(tweets)))
            return tweets

    

    def __train_data(self,tweets_filename,labels_filename):
        """
            train model with exists files:
            'tweets_filename','labels_filename'
        """

        tweet_texts = [ tweet['text'] for tweet in self.load_tweets(tweets_filename)]
        labels = self.load_laebls(labels_filename)
        n_samples = max(len(tweet_texts),len(labels))
        # train data : labels and tweets
        tweet_texts = [tx.lower() for tx in tweet_texts[:n_samples]]
        labels = labels[:n_samples]
        # print(len(tweet_texts)," texts")
        # print(len(labels)," labels")
        X_samples = list(set(tweet_texts))
        y_true = []
        
        # strip same tweets_text
        for text in X_samples:
            index = tweet_texts.index(text)
            y_true.append(labels[index])

        y_true = np.array(y_true)
        print('------------test model scrore------------')
        pipeline = Pipeline([('bag-of-words',NLTKBOW()),
                         ('vectorizer',DictVectorizer()),
                         ('naive-bayes',BernoulliNB())])
        tweets_scores = cross_val_score(pipeline,X_samples,y_true,cv=10,scoring='f1')
        print("model score :{0}".format(tweets_scores.mean()))
        return (X_samples,y_true)

    def get_model(self,tweets_filename,labels_filename,model_filename):
        """
            export model
        """
        tweets_text,labels = self.__train_data(tweets_filename=tweets_filename,
            labels_filename=labels_filename)
        pipeline = Pipeline([('bag-of-words',NLTKBOW()),
                         ('vectorizer',DictVectorizer()),
                         ('naive-bayes',BernoulliNB())])
        model = pipeline.fit(tweets_text,labels)
        
        joblib.dump(model,model_filename)

