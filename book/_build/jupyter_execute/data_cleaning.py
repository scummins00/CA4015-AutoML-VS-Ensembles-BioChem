#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning
# The following notebook is for data cleaning and preparation. The dataset provided by {cite}`fitzgerald_morrin_holland_2021` represents GCMS analysis of VOCs from pure cultures of bacteria. The data is semi-structured in nature. It presents some challenges such as missing values. In the Excel file, the data obtained from the GCMS is presented in multiple formats, namely:
# 1. Long
# 2. Wide
# 
# Both sheets represent the same data. We will be working with the '**Wide**' dataset. This is because features represented as columns work better for Google's AutoML Tables. There are various other sheets available in the Excel, but these serve no purpose for our analysis.

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[2]:


raw = pd.read_excel("../data/FrontiersDS.xlsx", sheet_name="Wide", skiprows=3)


# In[3]:


raw.head()


# ## Null-Values
# In the given dataset, rows represent **species & strains** of bacterial micro-organisms. The columns represent individual chemical-compounds commonly found in the volatile organic compounds (VOCs). {cite:p}`fitzgerald2021` informs us that:
# * Cells with missing data represent a species-media specific combination in which the presence of that particular compound was never recorded.
# * Cells with the value 0 represent a species-media spcific combination in which the presence of that compound was found in some equivalent sample, but not this particular sample.
# 
# Because of this knowledge, it is difficult to understand what should be done with the missing values. According to the Google Cloud Platform documentation for ['Best Practices for creating training data'](https://cloud.google.com/automl-tables/docs/data-best-practices#avoid_missing_values_where_possible), it is best to avoid missing values where possible. Values can be left missing if the column is set to be nullable.
# 
# [**TPOT**](http://epistasislab.github.io/tpot/) is an Automatic Machine Learning package in Python. In this particular case, using TPOT will prove more beneficial to us and will allow us more control. As of *Version 0.9* TPOT supports sparse matrices with a new built-in TPOT configuration "TPOT sparse". So, for us to support the use of missing values, we must use this particular configuration.

# ## Encoding
# We must ensure that the target variable is also presented as an integer. To do this, we use SKLearns label encoder. This creates a 1 to 1 mapping between the target values and integers.

# In[8]:


le = preprocessing.LabelEncoder()
le.fit(raw.Strain)
list(le.classes_)


# In[9]:


raw.Strain = le.transform(raw.Strain)
raw.Strain.unique()


# In[27]:


raw.to_csv('../data/cleaned/long.csv', index=False)


# ## Seperate By Media
# Let's divide the dataset by media to perform per-media analysis of clusters 

# In[10]:


filled = raw.fillna(0)


# In[11]:


tsb = filled[filled['Samples '].str.contains("TSB")]
bhi = filled[filled['Samples '].str.contains("BHI")]
lb = filled[filled['Samples '].str.contains("LB")]


# ## Standardization
# We will be performing PCA for feature reduction. This will allow us to better cluster the data later on. The sklearn implimentation of PCA does not handle NaN values. We will let all NaN values equal 0 to perform PCA.

# In[12]:


tsb_features = tsb.iloc[:,3:]
bhi_features = bhi.iloc[:,3:]
lb_features = lb.iloc[:,3:]
full_features = filled.iloc[:,3:]

x1 = StandardScaler().fit_transform(tsb_features)
x2 = StandardScaler().fit_transform(bhi_features)
x3 = StandardScaler().fit_transform(lb_features)
x4 = StandardScaler().fit_transform(full_features)


# ## Principal Component Analysis

# In[13]:


#Now let's perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x4)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[15]:


#Let's rejoin the columns describing the data
pca_tsb = pd.concat([principalDf, tsb[['Species', 'Strain', 'Samples ']].reset_index(drop=True)], axis = 1)
pca_bhi = pd.concat([principalDf, bhi[['Species', 'Strain', 'Samples ']].reset_index(drop=True)], axis = 1)
pca_lb = pd.concat([principalDf, lb[['Species', 'Strain', 'Samples ']].reset_index(drop=True)], axis = 1)


# In[16]:


pca_full = pd.concat([principalDf, filled[['Species', 'Strain', 'Samples ']].reset_index(drop=True)], axis = 1)


# In[18]:


#Let's write out our datta
pca_tsb.to_csv('../data/cleaned/tsb_components.csv', index=False)
pca_bhi.to_csv('../data/cleaned/bhi_components.csv', index=False)
pca_lb.to_csv('../data/cleaned/lb_components.csv', index=False)


# In[20]:


pca_full.to_csv('../data/cleaned/full_components.csv', index=False)

