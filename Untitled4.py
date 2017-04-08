
# coding: utf-8

# In[176]:

import numpy as np
import pandas as pd


# In[177]:

import nltk
import numpy as np
import plotly    #Python Graphing Library
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
eng_stopwords = set(stopwords.words('english'))


# In[178]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[179]:

train.rename(columns ={'description_x':'question1','description_y':'question2','same_security':'is_similar'},inplace=True)
train.head()


# In[180]:

test.rename(columns={'description_x':'question1','description_y':'question2','same_security':'is_similar'},inplace=True)
test.head()


# In[181]:

is_sim = train['is_similar'].value_counts()
data = [go.Histogram(
            x=train['is_similar'].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Is Similar',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
    bargap=0.7,
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[182]:

allQues = pd.DataFrame(pd.concat([train['question1'], train['question2']]))
allQues.columns =["questions"]

allQues["no_of_words"] = allQues["questions"].apply(lambda x: len(str(x).split()))
allQues.head()


# In[183]:

# plotting histogram of number of occurrences of words
data = [go.Histogram(
            x=allQues["no_of_words"].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Number of words in the question',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[184]:

# plotting histogram of number of characters in the data set
allQues["no_of_chars"] = allQues["questions"].apply(lambda x: len(str(x)))    
data = [go.Histogram(
            x=allQues["no_of_chars"].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Number of characters in the question',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[185]:

# changing the words in qestion to lower case and tokenizing each word
def get_unigrams(que):
    return [word for word in word_tokenize(que.lower()) if word not in eng_stopwords]

## Finding the intersection between two series in pandas and return len.
def get_common_unigrams(row):
    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) ) 

def get_common_unigram_ratio(row):
    return float(row["unigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)

train["unigrams_ques1"] = train['question1'].apply(lambda x: get_unigrams(str(x)))
train["unigrams_ques2"] = train['question2'].apply(lambda x: get_unigrams(str(x)))
train["unigrams_common_count"] = train.apply(lambda row: get_common_unigrams(row), axis=1)
train["unigrams_common_ratio"] = train.apply(lambda row: get_common_unigram_ratio(row),axis=1)


# In[186]:

data = [go.Histogram(
            x=train["unigrams_common_count"].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Common unigrams count',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[187]:

def get_bigrams(que):
    return [ i for i in ngrams(que,2)]

def get_common_bigrams(row):
    return len( set(row['bigrams_ques1']).intersection(set(row['bigrams_ques2'])) )

def get_common_bigram_ratio(row):
    return float(row["bigrams_common_count"]) / max(len( set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"])) ),1)

train["bigrams_ques1"] = train["unigrams_ques1"].apply(lambda x: get_bigrams(x))
train["bigrams_ques2"] = train["unigrams_ques2"].apply(lambda x: get_bigrams(x))
train["bigrams_common_count"] = train.apply(lambda row: get_common_bigrams(row), axis=1)
train["bigrams_common_ratio"] = train.apply(lambda row: get_common_bigram_ratio(row), axis=1)


# In[188]:

data = [go.Histogram(
            x=train['bigrams_common_count'].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Common bigrams count',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[189]:

data = [go.Histogram(
            x=train['bigrams_common_ratio'].astype(str),
    )]
layout = go.Layout(
    xaxis=dict(
        title='Common bigrams ratio',
    ),
    yaxis=dict(
        title='Number of Occurrences',
    ),
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='required')


# In[190]:

def features(row):
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    out_list = []
    # get unigram features
    unigrams_que1 = [word for word in que1.lower().split() if word not in eng_stopwords]
    unigrams_que2 = [word for word in que2.lower().split() if word not in eng_stopwords]
    common_unigrams_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
    common_unigrams_ratio = float(common_unigrams_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))),1)
    out_list.extend([common_unigrams_len, common_unigrams_ratio])
    
    # get bigram features
    bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    out_list.extend([common_bigrams_len, common_bigrams_ratio])
    
    # get trigram features
    trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
    trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
    common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    out_list.extend([common_trigrams_len, common_trigrams_ratio])
    return out_list


# In[191]:

trainX = np.vstack( np.array(train.apply(lambda row: features(row), axis=1)) ) 
testX = np.vstack( np.array(test.apply(lambda row: features(row), axis=1)))
trainY = np.array(train["is_similar"])
# test_id = np.array(test["test_id"])
print train_X


# In[192]:

train_X_similar = trainX[trainY==1]
train_X_non_similar = trainX[trainY==0]
# adding extra train_X_non_similar terms to avoid biasing as we have more number of observations which has True output
trainX = np.vstack([train_X_non_similar, train_X_similar, train_X_non_similar, train_X_non_similar])
trainY = np.array([0]*train_X_non_similar.shape[0] + [1]*train_X_similar.shape[0] + [0]*train_X_non_similar.shape[0] + [0]*train_X_non_similar.shape[0])


# In[193]:

# Combine train_X and train_Y in order to preserve mapping btween them
combined_mat = []
for idx, row_x in enumerate(trainX):
    combined_mat.append(np.append(row_x, (trainY[idx])))
# Shuffle the matrix to avoid biasing of our model
np.random.shuffle(combined_mat)
total_entries = len(combined_mat)
partition = int(total_entries*0.8)
for_training = combined_mat[:partition]
for_validation = combined_mat[partition:]
# Seperating the train_X and train_Y
trainX = np.vstack([for_training[idx][:-1] for idx, row in enumerate(for_training)])
trainY = np.array([for_training[idx][-1] for idx, row in enumerate(for_training)])
# print(train_X, train_y)
valX = np.vstack([for_validation[idx][:-1] for idx, row in enumerate(for_validation)])
valY = np.array([for_validation[idx][-1] for idx, row in enumerate(for_validation)])
# print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)


# In[194]:

# Choosing our model - DecisionTreeClassification
model = DecisionTreeClassifier()
model = model.fit(trainX, trainY)
model.score(trainX, trainY)


# In[195]:

# This cell contains the validation code for dataset

predicted = model.predict(valX)
correct = 0
total = predicted.shape[0]
for idx, _ in enumerate(predicted):
    if predicted[idx] == valY[idx]:
        correct += 1
accuracy = (float(correct)/total)
print 'accuracy =',(accuracy)


# In[196]:

# Get predictions from generated model for test data set
predicted_from_test = model.predict(testX)
for idx, prediction in enumerate(predicted_from_test):
    print(test_X[idx], prediction)


# In[ ]:



