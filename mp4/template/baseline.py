"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
import collections

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
   
    #Considers words independently, ignores previous words/tags
    #Train Data: Count # of times word occurs with each tag. Ex "dog" is noun(20) times, verb(0) times, adj(3)
    #Test: For unseen words use tag that is most seen
    sol = []
    wordDic = {}        #tag: counter() -> {word: count} word: {tag:count}
    cnt = collections.Counter()         #count total number of tags for total 
    tagStartEnd = {'START', 'END'}
   
    
    #fill up dictionary
    for sentence in train:
        for word in sentence:
            if word[1] not in tagStartEnd:
                if word[0] not in wordDic:
                    wordDic[word[0]] = collections.Counter()      #first instance of tag, initialize to {word: 1}
                wordDic[word[0]].update({word[1]: 1})
                cnt.update({word[1]: 1})

    mostCommonTag = cnt.most_common(1)[0][0]

    #test data

    #assign words in test data a tag
    for sentence in test:           
        solSentence = []
        for word in sentence:              
            if word not in tagStartEnd:
                if word in wordDic:
                    solSentence.append((word, wordDic[word].most_common(1)[0][0]))
                else:
                    solSentence.append((word, mostCommonTag))
            else:
                solSentence.append((word, word))    #START, END
        sol.append(solSentence)

    return sol