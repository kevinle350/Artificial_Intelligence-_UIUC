# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.2, pos_prior=0.9,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    yhats = []
    
    

    #training set is a list of lists of words: [[hi, eat, cool, people], [ice, friend, ranch, water], [sad, libary, somi, lamp]]
    #Each list of words [hi, eat, cool, people] contains all the words in one review
    #   Each list of words contains all the words in one review, this means one review(pos/neg) describes this whole list of words
    #Labels have positive or negative reviews
    
    #need to find probabilities using bayes 
    #make a dic for word to store its pos and neg probabilities
    #use training set to learn individual probabilities
    dic = {} #{word: (negative_prob, positive_prob)}   
    posCount = 0
    negCount = 0
    wordDic = {}
    posWord = 0
    negWord = 0
           
    # wordDic[word][0] - count word occurence in negative 
    # wordDic[word][1] - count word occurence in positive  
    for i in range(len(train_set)): 
        for word in train_set[i]:
            if train_labels[i] == 1:
                posCount += 1
                if word not in wordDic: 
                    wordDic[word] = [0, 1]
                else:
                    wordDic[word][1] += 1
            else:
                negCount += 1
                if word not in wordDic: 
                    wordDic[word] = [1, 0]
                else:
                    wordDic[word][0] += 1


    for word in wordDic:
        posWord += wordDic[word][1]
        negWord += wordDic[word][0]
       
    for word in wordDic:    
        # #dic[word] = [wordDic[word][0]/negCount, wordDic[word][1]/posCount]
        # if wordDic[word][0] == 0:
        #     dic[word] = [0, math.log(wordDic[word][1] + laplace)-math.log(posCount+laplace*(posWord+1))]
        # elif wordDic[word][1] == 0:
        #     dic[word] = [math.log(wordDic[word][0] + laplace)-math.log(negCount+laplace*(negWord+1)), 0]
        # else:
        dic[word] = [math.log(wordDic[word][0] + laplace)-math.log(negCount+laplace*(negWord+1)), math.log(wordDic[word][1] + laplace)-math.log(posCount+laplace*(posWord+1))]



    for doc in tqdm(dev_set,disable=silently):
        #yhats.append(-1)
        revPos = math.log(pos_prior)
        revNeg = math.log(1-pos_prior) 
        for word in doc:
            if word not in dic:
                revPos = revPos + math.log(laplace/(posCount + laplace*(posWord + 1)))
                revNeg = revNeg + math.log(laplace/(negCount + laplace*(negWord + 1)))
            elif dic[word][0] == 0:
                revPos = revPos + dic[word][1]
                revNeg = revNeg +  math.log(laplace/(negCount + laplace*(negWord + 1)))
            elif dic[word][1] == 0:
                revPos = revPos + math.log(laplace/(posCount + laplace*(posWord + 1)))
                revNeg = revNeg + dic[word][0]
            else:
                revPos = revPos + dic[word][1]
                revNeg = revNeg + dic[word][0]

        if revPos > revNeg:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats



    # for listWord in dev_set:
    #     for word in listWord:
    #         if word not in dic:
    #             posTot = posTot + math.log(laplace/posCount)
    #             negTot = negTot +  math.log(laplace/negCount)
    #         else:
    #             posTot = posTot + dic[word][1]
    #             negTot = negTot + dic[word][0]
            
    #     revPos = posTot * pos_prior
    #     revNeg = negTot * (1-pos_prior)
           
    #     if revPos > revNeg:
    #         devLabels.append(1)
    #     else:
    #         devLabels.append(0)

    # return devLabels










# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.2, bigram_laplace=0.01, bigram_lambda=0.5,pos_prior=0.8, silently=False):

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    # for doc in tqdm(dev_set,disable=silently):
    #     yhats.append(-1)
    # return yhats

    bi_dic = {} #{word: (negative_prob, positive_prob)}   
    bi_posCount = 0
    bi_negCount = 0
    bi_wordDic = {}
    bi_posWord = 0
    bi_negWord = 0
           
    # wordDic[word][0] - count word occurence in negative 
    # wordDic[word][1] - count word occurence in positive  
    for i in range(len(train_set)): 
        for j in range(len(train_set[i]) - 1):
            pair = train_set[i][j] + " " + train_set[i][j+1]
            if train_labels[i] == 1:
                bi_posCount += 1
                if pair not in bi_wordDic: 
                    bi_wordDic[pair] = [0, 1]
                else:
                    bi_wordDic[pair][1] += 1
            else:
                bi_negCount += 1
                if pair not in bi_wordDic: 
                    bi_wordDic[pair] = [1, 0]
                else:
                    bi_wordDic[pair][0] += 1


    for pair in bi_wordDic:
        bi_posWord += bi_wordDic[pair][1]
        bi_negWord += bi_wordDic[pair][0]
       
    for pair in bi_wordDic:    
        # #dic[word] = [wordDic[word][0]/negCount, wordDic[word][1]/posCount]
        # if wordDic[word][0] == 0:
        #     dic[word] = [0, math.log(wordDic[word][1] + laplace)-math.log(posCount+laplace*(posWord+1))]
        # elif wordDic[word][1] == 0:
        #     dic[word] = [math.log(wordDic[word][0] + laplace)-math.log(negCount+laplace*(negWord+1)), 0]
        # else:
        bi_dic[pair] = [math.log(bi_wordDic[pair][0] + bigram_laplace)-math.log(bi_negCount+bigram_laplace*(bi_negWord+1)), math.log(bi_wordDic[pair][1] + bigram_laplace)-math.log(bi_posCount+bigram_laplace*(bi_posWord+1))]

    uni_dic = {} #{word: (negative_prob, positive_prob)}   
    uni_posCount = 0
    uni_negCount = 0
    uni_wordDic = {}
    uni_posWord = 0
    uni_negWord = 0
           
    # wordDic[word][0] - count word occurence in negative 
    # wordDic[word][1] - count word occurence in positive  
    for i in range(len(train_set)): 
        for word in train_set[i]:
            if train_labels[i] == 1:
                uni_posCount += 1
                if word not in uni_wordDic: 
                    uni_wordDic[word] = [0, 1]
                else:
                    uni_wordDic[word][1] += 1
            else:
                uni_negCount += 1
                if word not in uni_wordDic: 
                    uni_wordDic[word] = [1, 0]
                else:
                    uni_wordDic[word][0] += 1


    for word in uni_wordDic:
        uni_posWord += uni_wordDic[word][1]
        uni_negWord += uni_wordDic[word][0]
       
    for word in uni_wordDic:    
        # #dic[word] = [wordDic[word][0]/negCount, wordDic[word][1]/posCount]
        # if wordDic[word][0] == 0:
        #     dic[word] = [0, math.log(wordDic[word][1] + laplace)-math.log(posCount+laplace*(posWord+1))]
        # elif wordDic[word][1] == 0:
        #     dic[word] = [math.log(wordDic[word][0] + laplace)-math.log(negCount+laplace*(negWord+1)), 0]
        # else:
        uni_dic[word] = [math.log(uni_wordDic[word][0] + unigram_laplace)-math.log(uni_negCount+unigram_laplace*(uni_negWord+1)), math.log(uni_wordDic[word][1] + unigram_laplace)-math.log(uni_posCount+unigram_laplace*(uni_posWord+1))]

    
    for doc in tqdm(dev_set,disable=silently):
        #yhats.append(-1)
        uniRevPos = math.log(pos_prior)
        uniNevNeg = math.log(1-pos_prior)        
        revPos = math.log(pos_prior)
        revNeg = math.log(1-pos_prior)        

        for i in range(len(doc) - 1):
            pair = doc[i] + " " + doc[i+1]
            if pair not in bi_dic:
                revPos = revPos + math.log(bigram_laplace/(bi_posCount + bigram_laplace*(bi_posWord + 1)))
                revNeg = revNeg + math.log(bigram_laplace/(bi_negCount + bigram_laplace*(bi_negWord + 1)))
            elif bi_dic[pair][0] == 0:
                revPos = revPos + bi_dic[word][1]
                revNeg = revNeg + math.log(bigram_laplace/(bi_negCount + bigram_laplace*(bi_negWord + 1)))
            elif bi_dic[pair][1] == 0:
                revPos = revPos + math.log(bigram_laplace/(bi_posCount + bigram_laplace*(bi_posWord + 1)))
                revNeg = revNeg + bi_dic[pair][0]
            else:
                revPos = revPos + bi_dic[pair][1]
                revNeg = revNeg + bi_dic[pair][0]


        for word in doc:
            if word not in uni_dic:
                uniRevPos = uniRevPos + math.log(unigram_laplace/(uni_posCount + unigram_laplace*(uni_posWord + 1)))
                uniNevNeg = uniNevNeg + math.log(unigram_laplace/(uni_negCount + unigram_laplace*(uni_negWord + 1)))
            elif uni_dic[word][0] == 0:
                uniRevPos = uniRevPos + uni_dic[word][1]
                uniNevNeg = uniNevNeg + math.log(unigram_laplace/(uni_negCount + unigram_laplace*(uni_negWord + 1)))
            elif uni_dic[word][1] == 0:
                uniRevPos = uniRevPos + math.log(unigram_laplace/(uni_posCount + unigram_laplace*(uni_posWord + 1)))
                uniNevNeg = uniNevNeg + uni_dic[word][0]
            else:
                uniRevPos = uniRevPos + uni_dic[word][1]
                uniNevNeg = uniNevNeg + uni_dic[word][0]
                
        val = revPos * bigram_lambda + uniRevPos * ( 1 - bigram_lambda) > revNeg * bigram_lambda + uniNevNeg * (1-bigram_lambda)
        #val = uniRevPos > uniNevNeg
        if val:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
             



   

    
