import re
import sys
import os

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    f = open(corpus_path,'r')
    data = f.read()
    data = data.split('\n')
    corpus = []
    for line in data:
        try:
            sent,label = line.split('\t')
            sent = sent.split()
            corpus.append((sent, int(label)))
        except:
            pass
    f.close()
    return corpus

# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words:
        return True
    if "n't" in word:
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    tagged = nltk.pos_tag(snippet)
    do_negation = False
    result = []
    prev = ''
    for i in snippet:
        if i == 'only' and prev == 'not':
            do_negation = False
        if i in sentence_enders or i in negation_enders or (i,'RBR') in tagged or (i,'JJR') in tagged:
            do_negation = False
        if do_negation:
            result.append('NOT_'+i)
        else:
            result.append(i)
        if is_negation(i):
            do_negation = True
            prev = i
    return result

# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    feature = dict()
    index = 0
    for sent,label in corpus:
        for w in sent:
            if w not in feature:
                feature[w]=index
                index+=1
    return feature
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    vector = np.zeros(len(feature_dict))
    for w in snippet:
        if w in feature_dict:
            vector[feature_dict[w]]+=1
        else:
            continue
    return vector


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X = np.empty((len(corpus), len(feature_dict)))
    Y = np.empty(len(corpus))
    for i,(sent,label) in enumerate(corpus):
        X[i] = vectorize_snippet(sent, feature_dict)
        Y[i] = int(label)
    return tuple([X,Y])

# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    for i in range(len(X[0])):
        xi = X[:,i]
        minx = np.min(xi)
        maxx = np.max(xi)
        if minx == maxx:
            minx = 0
        if maxx == 0:
            continue
        X[:,i] = (xi-minx)/(maxx-minx)
    return X


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    for i,(sent,label) in enumerate(corpus):
        tag = tag_negation(sent)
        corpus[i] = (sent,label)
    feature = get_feature_dictionary(corpus)
    X,Y = vectorize_corpus(corpus, feature)
    X = normalize(X)
    model = LogisticRegression()
    model.fit(X,Y)
    return (model, feature)

# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp,fp,fn = 0,0,0
    for i,yi in enumerate(Y_test):
        if yi==1 and Y_pred[i]==1:
            tp+=1
        elif yi==1 and Y_pred[i]==0:
            fn+=1
        elif yi==0 and Y_pred[i]==1:
            fp+=1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return (precision, recall, f1_score)


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    for i,(sent,label) in enumerate(corpus):
        tag = tag_negation(sent)
        corpus[i] = (sent,label)
    X_test,Y_test = vectorize_corpus(corpus, feature_dict)
    X_test = normalize(X_test)
    Y_pred = model.predict(X_test)
    return evaluate_predictions(Y_pred, Y_test)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    coef = logreg_model.coef_.reshape(-1)
    wt = []
    for i,co in enumerate(coef):
        wt.append([co,i])
    wt = sorted(wt)[::-1]
    result = []
    words = list(feature_dict.keys())
    for i in range(k):
        co,i = wt[i]
        result.append((words[i],co))
    return result


def main(args):
    lc = load_corpus('train.txt')
    print(lc[:2])
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
