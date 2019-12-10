#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#pip install keras-bert
#pip install tensorflow-gpu >=1.14
#pip install keras

import os
from keras import backend as K
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras_bert import Tokenizer
import keras
import re
import sys
import codecs
#from keras_radam import RAdam
pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# In[ ]:


#from keras_radam import RAdam


# In[ ]:


os.environ['TF_KERAS'] = '1'


# In[ ]:


token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# In[ ]:


from keras_bert import load_trained_model_from_checkpoint

model = load_trained_model_from_checkpoint(config_path, checkpoint_path)


# In[ ]:


f = open(sys.argv[1], 'r')
texts=[]
while True:
    line = f.readline()
    if not line: break
    line=re.sub('<br','',line)
    line=re.sub('/><br','',line)
    texts.append(line.rstrip('\n').lower())
    #texts.append(line)
f.close()


# In[ ]:


tokenizer = Tokenizer(token_dict)
n=0
for i in texts:
    tokens = tokenizer.tokenize(i)
    indices, segments = tokenizer.encode(first=i, max_len=512)
    predicts = model.predict([np.array([indices]), np.array([segments])])
    if n ==0:
        C=predicts
    else:
        C=np.vstack((C,predicts))
    if n>2000:
        print(n)
    n+=1


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
labels=np.asarray(labels,dtype='float32')



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


tokenizer = Tokenizer(token_dict)
n=0
for i in tests:
    tokens = tokenizer.tokenize(i)
    indices, segments = tokenizer.encode(first=i, max_len=512)
    predicts = model.predict([np.array([indices]), np.array([segments])])
    if n ==0:
        Test=predicts
    else:
        Test=np.vstack((Test,predicts))
    if n>2000:
        print(n)
    n+=1


# In[ ]:


keras.backend.clear_session() 


# In[ ]:


#keep it

unit=16
_input = keras.layers.Input(shape=(512,768), dtype='float')

'''# get the embedding layer
embedded = keras.layers.Embedding(max_words,emdim,weights=([embed_mtx]),trainable=False)(_input)
#conv1
conv1=keras.layers.Conv1D(5,1,activation='relu')(embedded)
max1=keras.layers.MaxPooling1D(3)(conv1)
#conv2
conv2=keras.layers.Conv1D(5,2,activation='relu')(embedded)
max2=keras.layers.MaxPooling1D(3)(conv2)
#conv3
conv3=keras.layers.Conv1D(5,3,activation='relu')(embedded)
max3=keras.layers.MaxPooling1D(3)(conv3)'''

#concat
#concat=keras.layers.Concatenate(axis=-1)([max1,max2,max3])
activations1 = keras.layers.Bidirectional(keras.layers.GRU(units=unit,
                                                       return_sequences=True,recurrent_dropout=0.3,
                                                         ))(_input)

# attention
attention1=keras.layers.Dense(1, activation='tanh')( activations1 )
attention1=keras.layers.Flatten()( attention1 )
attention1=keras.layers.Activation('softmax')( attention1 )
attention1=keras.layers.RepeatVector(unit*2)( attention1 )
attention1=keras.layers.Permute([2, 1])( attention1 )
activations1 = keras.layers.Multiply()([activations1, attention1])
activations1 = keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(unit*2,))(activations1)

activations2 = keras.layers.Bidirectional(keras.layers.GRU(units=unit,
                                                       return_sequences=True,recurrent_dropout=0.3,
                                                         ))(_input)

# attention2
attention2=keras.layers.Dense(1, activation='tanh')( activations2 )
attention2=keras.layers.Flatten()( attention2 )
attention2=keras.layers.Activation('softmax')( attention2 )
attention2=keras.layers.RepeatVector(unit*2)( attention2 )
attention2=keras.layers.Permute([2, 1])( attention2 )
activations2 = keras.layers.Multiply()([activations2, attention2])
activations2 = keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(unit*2,))(activations2)


'''
activations3 = keras.layers.Bidirectional(keras.layers.GRU(units=unit,
                                                       return_sequences=True,recurrent_dropout=0.3,
                                                         ))(max3)

# attention3
attention3=keras.layers.Dense(1, activation='tanh')( activations3 )
attention3=keras.layers.Flatten()( attention3 )
attention3=keras.layers.Activation('softmax')( attention3 )
attention3=keras.layers.RepeatVector(unit*2)( attention3 )
attention3=keras.layers.Permute([2, 1])( attention3 )
activations3 = keras.layers.Multiply()([activations3, attention3])
activations3 = keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(unit*2,))(activations3)

activations4 = keras.layers.Bidirectional(keras.layers.GRU(units=unit,
                                                       return_sequences=True,recurrent_dropout=0.3,
                                                         ))(embedded)

# attention4
attention4=keras.layers.Dense(1, activation='tanh')( activations4 )
attention4=keras.layers.Flatten()( attention4 )
attention4=keras.layers.Activation('softmax')( attention4 )
attention4=keras.layers.RepeatVector(unit*2)( attention4 )
attention4=keras.layers.Permute([2, 1])( attention4 )
activations4 = keras.layers.Multiply()([activations4, attention4])
activations4 = keras.layers.Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(unit*2,))(activations4)

'''

concat=keras.layers.Concatenate(axis=-1)([activations1,activations2])

#dense=keras.layers.Dense(64)(concat)
probabilities = keras.layers.Dense(1, activation='sigmoid')(concat)

model = Model(input=_input, output=probabilities)
adam = optimizers.Adam(lr=0.001, clipnorm=1.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])


# In[ ]:


history=model.fit(C,labels,epochs=50,batch_size=1800,validation_split=0.1,verbose=1) #45


# In[ ]:


PRED=model.predict(Test)


# In[ ]:


f = open(sys.argv[4], 'w')
for i in PRED:
    if i>=0.5:
        f.write('pos')
    elif i<0.5:
        f.write('neg')
    f.write('\n')
f.close()

