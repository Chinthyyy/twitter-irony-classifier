import nltk 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import warnings 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_data = pd.read_csv('train.txt', sep='\t')
test_data = pd.read_csv('test.csv')
final_data = train_data.append(test_data, ignore_index=True)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt
	
#preprocessing
final_data['processed_tweet'] = np.vectorize(remove_pattern)(final_data['Tweet text'], "@[\w]*")
final_data['processed_tweet'] = final_data['processed_tweet'].str.replace("[^a-zA-Z#]", " ")
final_data['processed_tweet'] = final_data['processed_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#tokenizing 
tokenized_tweet = final_data['processed_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

final_data['processed_tweet'] = tokenized_tweet

#creating bag of words
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(final_data['processed_tweet'])

train_bow = bow[:3817,:]
test_bow = bow[3817:,:]

#Data splitting
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_data['Label'], random_state=42, test_size=0.3)

clf = SVC(gamma='auto', probability=True)
clf.fit(xtrain_bow, ytrain)

prediction = clf.predict_proba(xvalid_bow) 
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)

test_pred = clf.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test_data['Label'] = test_pred_int

submission = test_data[['Tweet index','Label']]
submission.to_csv('sub_svm_bow.csv', index=False)

