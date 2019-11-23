# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:48:31 2019

@author: WHY
"""
import os
from collections import Counter

def run_main():
    print("Naive-Bayes Classifier")
    
    """
    Remove the document identifier and use 80% of whole doc for training.
    Remaining 20% would be for testing.
    Docs = The review sentences
    Labels = Sentiment (Positive or Negative)
    """
    
    currentDir = os.getcwd()
    theFile = currentDir + "\\all_sentiment_shuffled.txt"
    print("The file path: " + theFile)

    all_docs, all_labels = read_documents(theFile)
    #print(all_docs)
    #print(all_labels)
    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]
    
    
    """
    Makes a dictionary object of how many time a word appears. The format shown below
    word : 423
    The code shown below works better for all_labels
    """
    freqs3 = Counter(all_labels)    
    print(freqs3)
    
    
    """
    The code below works better for all_docs
    """
    freqs2 = Counter()
    for doc in all_docs:
        freqs2.update(doc)
    
    
    words_in_vocabulary, words_in_pos, words_in_neg, total_vocabulary = train_nb(train_docs, train_labels)
    
"""
Reading the document.
"""
def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


def train_nb(documents, labels):
    
    words_in_vocabulary = Counter()
    words_in_pos = []                   
    words_in_neg = []                  
    total_vocabulary = 0
    
    
    """
    This will track all vocabulary in the training set.
    Use this to check whether the word inputted by User is in the training set vocabulary
    """
    for word in documents:
        words_in_vocabulary.update(word)
    
    
    for i in range(len(documents)):
        if(labels[i] == "pos"):
            words_in_pos.append(documents[i])
    
    for i in range(len(documents)):
        if(labels[i] == "neg"):
            words_in_neg.append(documents[i])
    
    """
    Positive review vocabulary and their frequency
    """ 
    word_pos = Counter()
    for doc in words_in_pos:
        word_pos.update(doc)
    #print (word_pos)
    
    """
    Negative review vocabulary and their frequency
    """
    word_neg = Counter()
    for doc in words_in_neg:
        word_neg.update(doc)
    
    """
    Was thinking of adding smoothing directly, but I realised that for the 
    smoothing to work, it needs to be when the user enters a sentence with a
    word that does not exist in either "POS" or "NEG" set. That woule be the moment
    to add a 0.5 smoothing IF the word appears in the words_in_vocabulary.
    
    I think this would be better to implemented in the Part 2 because of how the
    Score formula is used.
    
    """
#    for key in word_pos:
#        value = word_pos.get(key)
#        value += 0.5
#        word_pos[key] = value
#    print(word_pos)   
    #print(positive_review)
    
    total_vocabulary = len(words_in_vocabulary.keys())
    
    return words_in_vocabulary, words_in_pos, words_in_neg, total_vocabulary
       



run_main()