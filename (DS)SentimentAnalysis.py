#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import required libraries
import pandas as pd


# In[3]:


#get the sentiment dataset
df_sentiment = pd.read_csv("C:\\Users\\91808\\Downloads\\imdb_labelled.txt",sep='\t',names=['comment','label'])


# In[4]:


#view first 10 observation 
# 1 indicates positive sentiment and 0 indicate negative sentiment
df_sentiment.head(10)


# In[5]:


#view more information about the sentiment data using describe method
df_sentiment.describe()


# In[6]:


#view more info on data
df_sentiment.info()


# In[7]:


#view the data using group by and describe method
df_sentiment.groupby('label').describe()


# In[8]:


#verify length of the message and also add it also as a new column (feature)
df_sentiment['length'] = df_sentiment['comment'].apply(len)


# In[9]:


#view first five message with length
df_sentiment.head()


# In[10]:


#view first link comment greater than 50 and index position zero(0)
df_sentiment[df_sentiment['length']>50]['comment'].iloc[0]


# In[11]:


#start text processing with vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[12]:


#define function to get rid of stopwords present in the messages
def message_text_process(mess):
    #check character to see if there are punctuations
    no_punctuation = [char for char in mess if char not in string.punctuation]
    #now form the sentence 
    no_punctuation = ''.join(no_punctuation)
    #Now eliminate any stopwords 
    return[word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


# In[13]:


#bag of words by applying the function and fit the data (comment) into it
import string
from nltk.corpus import stopwords
bag_of_words = CountVectorizer(analyzer=message_text_process).fit(df_sentiment['comment'])


# In[14]:


#apply transform method for the bag of words 
comment_bagofwords = bag_of_words.transform(df_sentiment['comment'])


# In[15]:


# apply tfidf transformer and fit the bag of words into it(transform version)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(comment_bagofwords)


# In[16]:


#print shape of the tfidf
comment_tfidf = tfidf_transformer.transform(comment_bagofwords)
print(comment_tfidf.shape)


# In[17]:


#choose naive bayes model  to detect the spam and fit the tfidf data into  it
from sklearn.naive_bayes import MultinomialNB
sentiment_detection_model = MultinomialNB().fit(comment_tfidf,df_sentiment['label'])


# In[23]:


#check model for the predicted and expected value say for comment# 1 and comment#5
comment = df_sentiment['comment'][4]
bag_of_words_for_comment = bag_of_words.transform([comment])
tfidf = tfidf_transformer.transform(bag_of_words_for_comment)
print('predicted sentiment label',sentiment_detection_model.predict(tfidf)[0])
print('expected sentiment label',df_sentiment.label[4])


# In[ ]:




