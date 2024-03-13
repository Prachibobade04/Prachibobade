#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import classification_report  
import re  
import string  


# # Import dataset

# In[2]:


fake = pd.read_csv("Fake.csv")  
true = pd.read_csv("True.csv")  


# In[3]:


fake


# In[4]:


fake.head()


# In[5]:


fake.info()


# In[6]:


fake.tail()


# In[7]:


true


# In[8]:


true.head()  


# In[9]:


true.info()


# In[10]:


true.tail()


# In[11]:


true["class"] = 1 
fake["class"] = 0 


# In[12]:


fake.shape, true.shape  


# In[13]:


fake


# In[14]:


true


# In[15]:


fake.shape


# In[16]:


true.shape


# # Merging True and Fake Dataframes

# In[17]:


dataframe_merge = pd.concat([fake, true], axis =0 )  
dataframe_merge.head(10)  


# In[18]:


dataframe_merge


# In[19]:


# We will remove the columns that are not required for us  
dataframe = dataframe_merge.drop(["title", "subject","date"], axis = 1)  
  
# Let's check if there are any null values in the dataset  
dataframe.isnull().sum()  


# In[20]:


# Here is the random shuffling of the rows in dataset   
data = dataframe.sample(frac=1) 
data.reset_index(inplace=True) 
data.drop(["index"], axis=1, inplace=True) 


# In[21]:


data


# In[22]:


sns.countplot(data=data, 
              x='class', 
              order=data['class'].value_counts().index)


# In[23]:


from tqdm import tqdm 
import re 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 


# In[24]:


def wordopt(t):
    t = t.lower()
    t = re.sub('\[.*?\]', '', t)
    t = re.sub("\\W"," ",t)
    t = re.sub('https?://\S+|www\.\S+', '', t)
    t = re.sub('<.*?>+', '', t)
    t = re.sub('[%s]' % re.escape(string.punctuation), '', t)
    t = re.sub('\n', '', t)
    t = re.sub('\w*\d\w*', '', t)    
    return t





dataframe["text"] = dataframe["text"].apply(wordopt)


# In[25]:


pip install WordCloud


# In[25]:


from wordcloud import WordCloud


# In[26]:


# True
consolidated = ' '.join( 
    word for word in data['text'][data['class'] == 1].astype(str)) 
wordCloud = WordCloud(width=1600, 
                      height=800, 
                      random_state=21, 
                      max_font_size=110, 
                      collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
plt.axis('off') 
plt.show() 


# In[27]:


# Fake 
consolidated = ' '.join( 
    word for word in data['text'][data['class'] == 0].astype(str)) 
wordCloud = WordCloud(width=1600, 
                      height=800, 
                      random_state=21, 
                      max_font_size=110, 
                      collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
plt.axis('off') 
plt.show() 


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer 
  
  
def get_top_n_words(corpus, n=None): 
    vec = CountVectorizer().fit(corpus) 
    bag_of_words = vec.transform(corpus) 
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()] 
    words_freq = sorted(words_freq, key=lambda x: x[1], 
                        reverse=True) 
    return words_freq[:n] 
  
  
common_words = get_top_n_words(data['text'], 20) 
df1 = pd.DataFrame(common_words, columns=['Review', 'count']) 
  
df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot( 
    kind='bar', 
    figsize=(10, 6), 
    xlabel="Top Words", 
    ylabel="Count", 
    title="Bar Chart of Top Words Frequency"
) 


# # Converting text into Vectors

# In[29]:


#Before converting the data into vectors, split it into train and test.
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
  
#Now we will define the dependent variable and independent variables  
x = dataframe["text"]  
y = dataframe["class"]  
  
# Splitting the Dataset into a Training and Testing Set  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer  
  
  
vectorization = TfidfVectorizer()  
xv_train = vectorization.fit_transform(x_train)  
xv_test = vectorization.transform(x_test)  


# # Model training, Evaluation, and Prediction

# # Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression  
  
  
LR = LogisticRegression()  
LR.fit(xv_train,y_train)  


# In[32]:


pred_lr=LR.predict(xv_test)  
LR.score(xv_test, y_test)  


# In[33]:


print(classification_report(y_test, pred_lr))  


# # Decision tree classifier

# In[34]:


from sklearn.tree import DecisionTreeClassifier  
  
  
DT = DecisionTreeClassifier()  
DT.fit(xv_train, y_train)  


# In[35]:


pred_dt = DT.predict(xv_test)  
DT.score(xv_test, y_test)  


# In[36]:


print(classification_report(y_test, pred_dt))  


# # model testing
#   we are going to use two models to check whether they are capable of detecting fake news. We have to check manually.

# In[41]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    else:
        return "Not A Fake News"  
     
def manual_testing(news):  
    testing_news = {"text":[news]}  
    new_def_test = pd.DataFrame(testing_news)  
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  
    new_x_test = new_def_test["text"]  
    new_xv_test =  vectorization.transform(new_x_test)  
    pred_LR = LR.predict(new_xv_test)  
    
    
  
  
    return print("\n\nLR Prediction: {} ".format(output_lable(pred_LR[0])))
  
news = str(input())  
manual_testing(news) 


# In[42]:


news = str(input())  
manual_testing(news) 


# In[ ]:




