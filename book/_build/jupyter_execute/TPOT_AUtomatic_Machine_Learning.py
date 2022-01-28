#!/usr/bin/env python
# coding: utf-8

# # TPOT
# In this notebook, we will define our TPOT pipeline, fit it to our training data and then use it to predict our test data.

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
train_data = pd.read_csv('../data/cleaned/long.csv')


# In[4]:


#Extract our training labels
train_labels = train_data.Strain

#Extract our training
train_features = train_data.drop(["Species", "Strain", "Samples "], axis=1)


# In[5]:


#Convert to 'numpy' arrays
training_features = np.array(train_features)
training_labels = np.array(train_labels).reshape((-1,))


# In[6]:


#Build the TPOT framework
tpot = TPOTClassifier(scoring = 'neg_mean_absolute_error',
                      max_time_mins = 500,
                      config_dict='TPOT sparse',
                      n_jobs=-1,
                      verbosity = 2,
                      cv=6)


# ## Fitting The Model
# The following cell will fit our TPOT model to our training data. It should be noted that this process is considerably faster with GPU utilisation. GPU utilisation is not a built-in feature with Jupyter Notebook. Therefore, this process is considerably faster on other platforms such as Google Colab.

# In[7]:


# Fit the tpot model on the training data
tpot.fit(training_features, training_labels)


# In[8]:


# Show the final model
print(tpot.fitted_pipeline_)


# Once we are finished with the model we export it to a file for use later on.

# In[9]:


# Export the pipeline as a python script file
tpot.export('tpot_exported_pipeline.py')

