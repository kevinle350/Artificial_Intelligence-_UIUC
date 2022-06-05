"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import collections

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    LAPLACE = 0.0000000001
    #print(train[0])
    #print(test[1])
    sol = []
    tagCount = {}        #counter for tags
    tagPairCount = {}     #counter for tag/word pairs
    wordTagCount = {}     #counter for words
    startTagCount = {}    #counter for start tag
    for sentence in train:
        for i in range(len(sentence)-1):    
            (w, t) = sentence[i]
            (nw, nt) = sentence[i+1]
            if i == 0:
                if w not in startTagCount:
                    startTagCount[w] = 0
                startTagCount[w] += 1
            if t not in tagCount:
                tagCount[t] = 0
            if t not in tagPairCount:
                tagPairCount[t] = {}          
            if nt not in tagPairCount[t]:
                tagPairCount[t][nt] = 0       
            if t not in wordTagCount: 
                wordTagCount[t] = {}         
            if w not in wordTagCount[t]:
                wordTagCount[t][w] = 0
            tagCount[t] += 1
            tagPairCount[t][nt] += 1
            wordTagCount[t][w] += 1
        
        (w, t) = sentence[-1]
        if t not in tagCount:
            tagCount[t] = 0
        if t not in wordTagCount:
            wordTagCount[t] = {}
        if w not in wordTagCount[t]:
            wordTagCount[t][w] = 0
        tagCount[t] += 1
        wordTagCount[t][w] += 1
    
    tagList = []
    for tag in tagCount.keys():
        tagList.append(tag)

    startN = sum(startTagCount.values())
    startV = len(startTagCount)
    ct = 1
    for word in test:
        ct += 1
        trellis = []
        b = {}
        for r in range(len(word)):
            trellis.append([])
            for c in range(len(tagList)):
                trellis[r].append(-math.inf)
        
        curRet = []
        for i in range(len(word)):
            w = word[i]
            if i == 0:
                for j in range(len(tagList)):
                    t = tagList[j]
                    if t not in startTagCount:
                        pS = math.log(LAPLACE/(startN+LAPLACE*(startV+1)))
                    else:
                        pS = math.log((startTagCount[t]+LAPLACE)/(startN+LAPLACE*(startV+1)))
                    
                    tagWordN = sum(wordTagCount[t].values())
                    tagWordV = len(wordTagCount[t])

                    if w not in wordTagCount[t]:
                        pE = math.log(LAPLACE / (tagWordN + LAPLACE * (tagWordV + 1)))
                    else:
                        pE = math.log((wordTagCount[t][w] + LAPLACE) / (tagWordN + LAPLACE * (tagWordV + 1)))
                    trellis[i][j] = pS + pE
            else:
                for j in range(len(tagList)):
                    t = tagList[j]
                    tagWordN = sum(wordTagCount[t].values())
                    tagWordV = len(wordTagCount[t])

                    if w not in wordTagCount[t]:
                        pE = math.log(LAPLACE / (tagWordN + LAPLACE * (tagWordV + 1)))
                    else:
                        pE = math.log((wordTagCount[t][w] + LAPLACE) / (tagWordN + LAPLACE * (tagWordV + 1)))
                    
                    maxIdx = 0
                    maxVal = -math.inf

                    for k, prevVal in enumerate(trellis[i-1]):
                        prevTag = tagList[k]
                        if prevTag not in tagPairCount:
                            continue
                        tagPairN = sum(tagPairCount[prevTag].values())
                        tagPairV = len(tagPairCount[prevTag])
                        if t not in tagPairCount[prevTag]:
                            pT = math.log(LAPLACE / (tagPairN + LAPLACE * (tagPairV + 1)))
                        else:
                            pT = math.log((tagPairCount[prevTag][t] + LAPLACE) / (tagPairN + LAPLACE * (tagPairV + 1)))
                        curVal = pT + prevVal
                        if curVal > maxVal:
                            maxVal = curVal
                            maxIdx = k
                    b[(i, j)] = (i-1, maxIdx)
                    trellis[i][j] = maxVal + pE

        lastMax = -math.inf
        lastIdx = 0
        for l, val in enumerate(trellis[-1]):
            if val > lastMax:
                lastMax = val
                lastIdx = l
        curRet.append((word[-1], tagList[lastIdx]))
        idxPair = (len(word)-1, lastIdx)
        idxPair = b[idxPair]
        while idxPair in b:
            wordIdx = idxPair[0]
            tagIdx = idxPair[1]
            curRet.insert(0, (word[wordIdx], tagList[tagIdx]))
            idxPair = b[idxPair]
        wordIdx = idxPair[0]
        tagIdx = idxPair[1] 
        curRet.insert(0, (word[wordIdx], tagList[tagIdx]))

        sol.append(curRet)        
    return sol   