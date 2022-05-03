#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction
# 
# **Based on dataset provided by kaggale. Dataset url.**

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
import math
import scipy.stats as stats
import scipy
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv("Bengaluru_House_Data (1).csv")
# data.head()


# In[5]:


data=data.drop(["availability","society","area_type","balcony"],axis=1)


# In[7]:


data=data.dropna()


# In[9]:


data.drop_duplicates(inplace=True)


# In[11]:


data['BHK']=data["size"].apply(lambda x: int(x.split(' ')[0]))


# In[14]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[15]:


# data[~data['total_sqft'].apply(isfloat)].head(10)


# In[16]:


def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[17]:


data=data.copy()
data['total_sqft']=data['total_sqft'].apply(convert_sqft_tonum)


# In[21]:


data=data.dropna()


# In[22]:


data.insert(4,"price_per_sqft",data.price*100000/data.total_sqft)


# In[23]:


data.price_per_sqft=data.price_per_sqft.round(2)


# In[25]:


location_stats=data.location.value_counts()


# In[27]:


locationlessthan10=location_stats[location_stats<=10]


# In[30]:


data.location=data.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
# len(data.location.unique())


# In[31]:


data=data[~(data.total_sqft/data.BHK<300)]
# data.head(10)


# In[34]:


data=data[data.bath<data.BHK+2]


# In[36]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
data=remove_pps_outliers(data)
# data.shape
# remove_pps_outliers(data)


# In[37]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

data=remove_bhk_outliers(data)
# data.shape


# In[38]:


data1=data.drop(['size','price_per_sqft'],axis='columns')
data=data.drop(['size','price_per_sqft'],axis='columns')
# data


# In[39]:


dummies=pd.get_dummies(data.location)
# dummies


# In[40]:


data=pd.concat([data,dummies.drop("other",axis="columns")],axis="columns")


# In[42]:


data=data.drop(['location'],axis=1)
# data


# In[43]:


X=data.drop(["price"],axis=1)
y=data["price"]


# In[44]:


np.random.seed(30)
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, 
                                                   y, 
                                                   test_size = 0.2)


# Ridge regression(Accuracy)

# In[46]:


from sklearn.linear_model import Ridge


# In[47]:


model2=Ridge()
model2.fit(X_train,y_train)
model2.score(X_test,y_test)


# In[48]:


def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0] 
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model2.predict([x])[0]


# In[49]:


# for content in data1.location:
#     print (content.value_counts())
data2 = sorted(list(set(data1.location)))
# print(data2)

for i in range (0,len(data2)):
    data2[i]=data2[i],i+1
#     print(data2[i])
    
# print(data2)


# In[50]:


data_bhk=sorted(data.BHK.unique().tolist())
for i in range(0,len(data_bhk)):
    data_bhk[i]=data_bhk[i],i+1
#     print(data_bhk[i])


# In[51]:


data_bath = sorted(data1.bath.astype(int).unique().tolist())
for i in range(0,len(data_bath)):
    data_bath[i]=data_bath[i],i+1
#     print(data_bath[i])


# In[52]:


import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
loc_dropdown = widgets.Dropdown(
    options=data2,    
    value=1,
    description='location',
)
bhk_dropdown= widgets.Dropdown(
    options=data_bhk,    
    value=1,
    description='BHK',
)
bath_dropdown= widgets.Dropdown(
    options=data_bath,    
    value=1,
    description='bath',
)
squarefeet_slider = widgets.IntSlider(
    val=0,
    min=0,
    max=15000,
    step=10,
    description='Square Feet',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
display(loc_dropdown)
display(bhk_dropdown)
display(bath_dropdown)
display(squarefeet_slider)


button = widgets.Button(description='Predict')
out = widgets.Output()
def on_button_clicked(_):
      # "linking function with output"
      with out:
          # what happens when we press the button
          clear_output()
          print('you cant predict tomorrow but you can predict housing rates!')
          print(price_predict(loc_dropdown.label,squarefeet_slider.value,bath_dropdown.value,bhk_dropdown.value))
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])


# # 

# In[53]:


import pickle

pickle.dump(model2,open("ridge_regression.pkl","wb"))


# In[54]:


loaded_model = pickle.load(open("random_forest_model_1.pkl","rb"))
# loaded_model.score(X_test,y_test)


# In[ ]:




