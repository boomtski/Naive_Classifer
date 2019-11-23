# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:48:31 2019

@author: WHY
"""

def run_main():
    print("Naive-Bayes Classifier")
    
    
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

"""
Remove the document identifier and use 80% of whole doc for training.
Remaining 20% would be for testing.
"""
