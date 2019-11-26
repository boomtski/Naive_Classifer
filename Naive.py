# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:48:31 2019

@authors: WHY, Ea10
"""
from __future__ import division
import os
from collections import Counter
import math
from codecs import open


def run_main():
    print('\nTask 0 and 1:\n')
    print("Naive-Bayes Classifier")
    
    """
    Remove the document identifier and use 80% of whole doc for training.
    Remaining 20% would be for testing.
    Docs = The review sentences
    Labels = Sentiment (Positive or Negative)
    """

    """
    Task 0 and 1
    """
    current_dir = os.getcwd()
    the_file = current_dir + "\\all_sentiment_shuffled.txt"
    print("\nThe file path: " + the_file)

    all_docs, all_labels = read_documents(the_file)
    # print(all_docs)
    # print(all_labels)
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
    print('\ntotal number of each label (100% of data): ', end='')
    print(freqs3)

    """
    The code below works better for all_docs
    """
    freqs2 = Counter()
    for doc in all_docs:
        freqs2.update(doc)

    dict_pos, dict_neg, log_prob_pos, log_prob_neg = train_nb(train_docs, train_labels)
    
    # -----------------------------------------------------------------------------------------------------

    """
    Task 2
    uncomment each test file for use as needed
    """
    print_review_content = True
    # test_file_1 = current_dir + "\\test_file_1.txt"
    # test_doc, test_label = read_documents(test_file_1)
    #
    # test_file_2 = current_dir + "\\test_file_2.txt"
    # test_doc, test_label = read_documents(test_file_2)

    # test_file_3 = current_dir + "\\test_file_3.txt"
    # test_doc, test_label = read_documents(test_file_3)
    
    test_file_task_4 = current_dir + "\\test_file_3.txt"
    test_doc, test_label = read_documents(test_file_task_4)

    # print(test_doc)
    # print(test_label)
    
    print('\nTask 2:\n')
    
    score_pos, score_neg = score_doc_label(test_doc, test_label, dict_pos, dict_neg, log_prob_pos, log_prob_neg,
                                           print_review_content)
    print('scores: ' + str(score_pos) + ', and ' + str(score_neg))

    guess = classify_nb(test_doc, score_pos, score_neg)

    guess_output = ''

    if guess == 'pos':
        guess_output = 'positive review! :)'
    elif guess == 'neg':
        guess_output = 'negative review! :('
    
    print('guess: ' + guess_output)
    
    # -----------------------------------------------------------------------------------------------------
    
    """
    Task 3
    """
    print_review_content = False
    list_of_guessed_labels = classify_documents(eval_docs, eval_labels, dict_pos, dict_neg, log_prob_pos, log_prob_neg,
                                                print_review_content)
    print('\nHere is the list for the prediction results for the last 20% of the reviews\n')
    print(list_of_guessed_labels)
    
    # calculate the accuracy value
    accuracy_value = accuracy(eval_labels, list_of_guessed_labels, eval_docs)
    print('\nThe accuracy of the program is: ', end='')
    print(accuracy_value)
    

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
    
    """
    This will track all vocabulary in the training set.
    Use this to check whether the word inputted by User is in the training set vocabulary
    """
    for word in documents:
        words_in_vocabulary.update(word)

    for i in range(len(documents)):
        if labels[i] == "pos":
            words_in_pos.append(documents[i])
        elif labels[i] == "neg":
            words_in_neg.append(documents[i])
            
    """
    Positive review vocabulary and their frequency
    """ 
    word_pos = Counter()
    for doc in words_in_pos:
        word_pos.update(doc)   
    """
    Negative review vocabulary and their frequency
    """
    word_neg = Counter()
    for doc in words_in_neg:
        word_neg.update(doc)
    # the update function for the counter will also count the number of words inside each sentence (each index)

    """
    Adding words that do not appear in Positive
    To handle all the words that did not appear in a category (pos or neg)
    If a word appeared in pos, but not in neg, it will appear as non-existant in the neg category.
    That is why we want to set it to 0.
    """
    dict_pos = word_pos
        
    for word in word_neg:
        if word not in word_pos:
            dict_pos[word] = 0
    # print(dict_pos)

    # Smoothing
    for word in dict_pos:
        value = dict_pos.get(word)
        value += 0.5
        dict_pos[word] = value
    # print(dict_pos)

    """
    Adding words that do not appear in Negative
    """
    dict_neg = word_neg
      
    for word in word_pos:
        if word not in word_neg:
            dict_neg[word] = 0
    # print(dict_neg)
    
    for word in dict_neg:
        value = dict_neg.get(word)
        value += 0.5
        dict_neg[word] = value
    # print(dict_neg)
    
    """
    Calculating the log probabilities of each word
    """
    total_pos_words = 0
    total_neg_words = 0
    
    for word in dict_pos:
        total_pos_words += dict_pos.get(word)
    print('total positive words: ', end='')
    print(total_pos_words)
    
    for word in dict_neg:
        total_neg_words += dict_neg.get(word)
    print('total negative words: ', end='')
    print(total_neg_words)

    vocab_size = sum(words_in_vocabulary.values())
    print("size of vocabulary:", end=' ')
    print(vocab_size)
    
    for word in dict_pos:
        probability = math.log(dict_pos.get(word) / (total_pos_words + 0.5 * vocab_size))
        dict_pos[word] = probability
    # print(dict_pos)

    for word in dict_neg:
        probability = math.log(dict_neg.get(word) / (total_neg_words + 0.5 * vocab_size))
        dict_neg[word] = probability
    # print(dict_neg)

    """
    Probability of Positive review and Negative review
    """
    reviews = Counter(labels)
    print('positive and negative reviews (80%): ', end='')
    print(reviews)
    
    total_labels = 0
    pos_labels = 0
    neg_labels = 0
    
    for review in reviews:
        if review == "pos":
            pos_labels += reviews.get(review)
        else:
            neg_labels += reviews.get(review)
        total_labels += reviews.get(review)
    
    print('total positive and negative reviews (80%): ', end='')
    print(total_labels)
    print('total positive reviews (80%): ', end='')
    print(pos_labels)
    print('total negative reviews (80%): ', end='')
    print(neg_labels)
    
    log_prob_pos = math.log(pos_labels / total_labels)
    log_prob_neg = math.log(neg_labels / total_labels)
    
    print('log probability of positive review:', end=' ')
    print(log_prob_pos)
    print('log probability of negative review:', end=' ')
    print(log_prob_neg)

    """
    dict_pos: list of words that appear in the positive vocabulary
    dict_neg: list of words that appear in the negative vocabulary
    log_prob_pos: probability of each word in the positive vocabulary
    log_prob_neg: probability of each word in the negative vocabulary
    """
    return dict_pos, dict_neg, log_prob_pos, log_prob_neg


"""
don't think 'label' is needed for this function??? But it's written on the project description
"""


def score_doc_label(document, label, dict_pos, dict_neg, log_prob_pos, log_prob_neg, print_review_content):

    score_pos = log_prob_pos
    score_neg = log_prob_neg

    if print_review_content is True:
        print('Now scoring document: \n')
        print('Printing document:')
        print('\n', document)

    # match each word in the document with the probability of its occurrence from the training data
    for word in document[0]:
        if word in dict_pos:
            score_pos += dict_pos.get(word)

    for word in document[0]:
        if word in dict_neg:
            score_neg += dict_neg.get(word)

    return score_pos, score_neg


"""
didn't have to use the 'document' parameter here... am I doing something wrong???
"""


def classify_nb(document, score_pos, score_neg):
    if score_pos > score_neg:
        guess = 'pos'
    else:
        guess = 'neg'

    return guess


"""
For task 3
"""


def classify_documents(eval_docs, eval_labels, dict_pos, dict_neg, log_prob_pos, log_prob_neg, print_review_content):
    predicted_sentiment_labels = []

    print("total positive reviews (20%):", end='')
    print(len(eval_docs))
    print("total negative reviews (20%):", end='')
    print(len(eval_labels))

    for i in range(len(eval_docs)):

        # Return the score for pos and neg for the last 20% of the reviews in the document
        score_pos_test, score_neg_test = score_doc_label(eval_docs[i], eval_labels[i], dict_pos, dict_neg, log_prob_pos, log_prob_neg, print_review_content)
        
        # Classify every document results one by one
        guess = classify_nb(eval_docs, score_pos_test, score_neg_test)
        
        # Store every guess result inside a list in order of review
        predicted_sentiment_labels.append(guess)
    
    return predicted_sentiment_labels


def accuracy(eval_labels, predicted_sentiment_labels, eval_docs):
    
    correctly_classified_counter = 0
    total_number_of_test_doc = len(eval_labels)
    misclassified_list = []
    
    # print('\nThe true labels:\n')
    # print(true_labels)

    for label in range(total_number_of_test_doc):
        if eval_labels[label] == predicted_sentiment_labels[label]:
            correctly_classified_counter += 1
        else:
            # Store all few misclassified documents in a list
            misclassified_list.append(eval_docs[label])
    
    # print('\nMisclassified Documents:\n')
    # print(misclassified_list)

    final_accuracy = correctly_classified_counter / total_number_of_test_doc
    
    return final_accuracy


run_main()
























