#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import numpy as np
import re
import sys


# In[ ]:


f = open(sys.argv[1], 'r')
texts=[]
while True:
    line = f.readline()
    if not line: break
    line=re.sub('<br','',line)
    line=re.sub('/><br','',line)
    texts.append(line.rstrip('\n').lower())
f.close()


# In[ ]:


f = open(sys.argv[2], 'r')
labels=[]
while True:
    line = f.readline()
    if not line: break
    if line.rstrip('\n')=='pos' :
        labels.append(1)
    elif line.rstrip('\n')=='neg' :
        labels.append(0)
f.close()


# In[ ]:


f = open(sys.argv[3], 'r')
tests=[]
while True:
    line = f.readline()
    if not line: break
    line=re.sub('<br','',line)
    line=re.sub('/><br','',line)
    tests.append(line.rstrip('\n').lower())
f.close()


# In[ ]:


def preprocess(texts):
    texts = texts.lower()
    words = word_tokenize(texts)
    words = [w for w in words if len(w) > 2]
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  
    w = []
    for i in range(len(words) - 1):
        #if i == len(words)-2:
            #break
        w += [' '.join(words[i:i + 2])]
    return w


# In[ ]:
pos_texts=0
neg_texts=0
for i in labels:
    if i ==1:
        pos_texts+=1
    elif i==0:
        neg_texts+=1
    else:
        print('error')
        break
        
tot_texts = pos_texts + neg_texts
pos_words = 0
neg_words = 0
tf_pos = dict()
tf_neg = dict()
idf_pos = dict()
idf_neg = dict()
#for i in range(texts.shape[0]):
for i in range(len(texts)):
    done_text = preprocess(texts[i])
    count = list() 
    for j in done_text:
        if labels[i] == 1 :
            tf_pos[j] = tf_pos.get(j, 0) + 1
            pos_words += 1
        elif labels[i] == 0:
            tf_neg[j] = tf_neg.get(j, 0) + 1
            neg_words += 1
        if j not in count:
            count += [j]
    for k in count:
        if labels[i] == 1 :
            idf_pos[k] = idf_pos.get(k, 0) + 1
        elif labels[i] == 0:
            idf_neg[k] = idf_neg.get(k, 0) + 1


# In[ ]:


#spam > pos
#ham > neg
prob_pos = dict()
prob_neg = dict()
sum_tf_idf_pos = 0
sum_tf_idf_neg = 0
for word in tf_pos:
    prob_pos[word] = (tf_pos[word]) * log((pos_texts + neg_texts) / (idf_pos[word] + idf_neg.get(word, 0)))
    sum_tf_idf_pos += prob_pos[word]
for word in tf_pos:
    prob_pos[word] = (prob_pos[word] + 1) / (sum_tf_idf_pos + len(list(prob_pos.keys())))

for word in tf_neg:
    prob_neg[word] = (tf_neg[word]) * log((pos_texts + neg_texts) / (idf_pos.get(word, 0) + idf_neg[word]))
    sum_tf_idf_neg += prob_neg[word]
for word in tf_neg:
    prob_neg[word] = (prob_neg[word] + 1) / (sum_tf_idf_neg + len(list(prob_neg.keys())))


prob_pos_texts, prob_neg_texts = pos_texts / tot_texts, neg_texts / tot_texts 


# In[ ]:


def classify(done_text):
    pos, neg = 0, 0
    for word in done_text:                
        if word in prob_pos:
            pos += log(prob_pos[word])
        else:
            pos -= log(sum_tf_idf_pos + len(list(prob_pos.keys())))
        if word in prob_neg:
            neg += log(prob_neg[word])
        else:
            neg -= log(sum_tf_idf_neg + len(list(prob_neg.keys()))) 
        pos += log(prob_pos_texts)
        neg += log(prob_neg_texts)
    return pos >= neg


# In[ ]:


def predict(data):
    result = dict()
    for (i, j) in enumerate(data):
        done_text = preprocess(j)
        result[i] = int(classify(done_text))
    return result


# In[ ]:


preds_tf_idf = predict(tests)


# In[ ]:


f = open(sys.argv[4], 'w')
for i in preds_tf_idf:
    if preds_tf_idf[i]==1:
        f.write('pos')
    elif preds_tf_idf[i]==0:
        f.write('neg')
    f.write('\n')
f.close()
