#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import sys


# In[2]:


dataset = pd.read_csv('quora_duplicate_questions.tsv', delimiter="\t")


# In[3]:


dataset.head()


# In[5]:


for i  in  range(len(dataset.columns.values)):
    print (i, dataset.columns.values[i], end = " ")
    print()


# In[6]:


X_temp = dataset.iloc[:, 3:5].values
y_temp = dataset.iloc[: , 5].values


# In[10]:


character_threshold = 100
X_thres_question_1 = []
X_thres_question_2 = []
y_thres = []
for i in range(len(X_temp)):
    if type(X_temp[i][0]) == str and type(X_temp[i][1]) == str and len(X_temp[i][0]) <= 100 and len(X_temp[i][1]) <= 100:
        X_thres_question_1.append(X_temp[i][0])
        X_thres_question_2.append(X_temp[i][0])
        y_thres.append(y_temp[i])


# In[11]:


len(X_thres_question_1)


# In[12]:


X_temp.shape


# In[13]:


X_thres_question_1[0]


# ## TEXT PREPROCESSING

# In[15]:


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import string
nltk.download('stopwords')


# In[16]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\(\[]*?[\)\]]", "", text)
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in stripped]
    words = [word.lower() for word in stemmed]
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    text = ' '.join(words)
    return text


# In[25]:


X_question_1_cleaned = []
X_question_2_cleaned = []

for i in range(len(X_thres_question_1)):
    sys.stdout.write('\rProcessing %d' %(i+1))
    sys.stdout.flush()
    X_question_1_cleaned.append(clean_text(X_thres_question_1[i]))
    X_question_2_cleaned.append(clean_text(X_thres_question_2[i]))


# In[26]:


X_question_1_cleaned[0]


# In[27]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint


# In[46]:


vocab = set()
for i in range(len(X_question_1_cleaned)):
    for j in (X_question_1_cleaned[i].split(' ')):
        vocab.add(j)
    for j in (X_question_2_cleaned[i].split(' ')):
        vocab.add(j)
print(len(vocab))


# In[37]:


import pickle


# In[47]:


with open('vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# In[48]:


word2int = dict()
vocab = list(vocab)
for i in range(len(vocab)):
    word2int[vocab[i]] = i


# In[49]:


word2int['<UNK>'] = len(word2int)


# In[50]:


with open('word2int.pickle', 'wb') as handle:
    pickle.dump(word2int, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[53]:


X_1 = []
X_2 = []
for i in range(len(X_question_1_cleaned)):
    _X_tmp_1 = []
    for j in (X_question_1_cleaned[i].split(' ')):
        _X_tmp_1.append(word2int[j])
    X_1.append(_X_tmp_1)
    _X_tmp_2 = []
    for j in (X_question_2_cleaned[i].split(' ')):
        _X_tmp_1.append(word2int[j])
    X_2.append(_X_tmp_2)


# In[55]:


max_length = -1
for i in range(len(X_1)):
    max_length = max(max_length, max(len(X_1[i]), len(X_2[i])))


# In[56]:


max_length


# In[57]:


X_1 = pad_sequences(X_1, max_length)
X_2 = pad_sequences(X_2, max_length)


# In[64]:


vocab_size = len(vocab) + 2


# In[114]:


y = np.array(y_thres)


# In[102]:


from keras.layers import Merge, Reshape, Flatten


# # TRAINING

# In[130]:


question_1_model = Sequential()
question_1_model.add(Embedding(vocab_size, 64, input_length=X_1.shape[1]))
#question_1_model.add(Dense(128, activation='relu'))
#question_1_model.add(Dense(64, activation='relu'))

question_2_model = Sequential()
question_2_model.add(Embedding(vocab_size, 64, input_length=X_1.shape[1]))
#question_2_model.add(Dense(128, activation='relu'))
#question_2_model.add(Dense(64, activation='relu'))

model = Sequential()
model.add(Merge([question_1_model, question_2_model], mode='concat'))
model.add(Flatten())
model.add(Dense(64, activation='relu', init='uniform'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics = ['acc'], optimizer='adam')


# In[131]:


model.summary()


# In[132]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y, 
                                                            random_state = 0, 
                                                            test_size = 0.1)
print(y_test_1[1:15])
X_train_2, X_test_2, y_train_1, y_test_1 = train_test_split(X_2, y, 
                                                            random_state = 0, 
                                                            test_size = 0.1)
print(y_test_1[1:15])


# In[133]:


model.fit([X_train_1, X_train_2], y_train_1, 
          validation_data=[[X_test_1, X_test_2], y_test_1], 
          epochs = 3, 
          batch_size = 128)


# In[134]:


from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

