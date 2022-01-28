#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install TPOT
get_ipython().system('pip install tpot')


# In[2]:


#Let's import our remaining packages
import pandas as pd
import numpy as np

# Import the tpot regressor
from tpot import TPOTClassifier


# In[3]:


#Read in our data (If above not working) you must import the file yourself.
#File will be deleted once GPU runtime expires.
train_data = pd.read_csv('data/cleaned/long.csv')


# In[7]:


#Extract our training labels
train_labels = train_data.Strain

#Extract our training
train_features = train_data.drop(["Species", "Strain", "Samples "], axis=1)


# In[8]:


#Convert to 'numpy' arrays
training_features = np.array(train_features)
training_labels = np.array(train_labels).reshape((-1,))


# In[9]:


#Build the TPOT framework
tpot = TPOTClassifier(scoring = 'neg_mean_absolute_error',
                      max_time_mins = 500,
                      config_dict='TPOT sparse',
                      n_jobs=-1,
                      verbosity = 2,
                      cv=6)


# In[11]:


# Fit the tpot model on the training data
tpot.fit(training_features, training_labels)


# In[ ]:


# Show the final model
print(tpot.fitted_pipeline_)


# In[ ]:


# Export the pipeline as a python script file
tpot.export('tpot_exported_pipeline.py')


# In[ ]:




