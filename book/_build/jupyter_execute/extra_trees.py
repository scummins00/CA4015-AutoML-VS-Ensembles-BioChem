#!/usr/bin/env python
# coding: utf-8

# # Extra Trees
# In the following Jupyter Notebook, we will be using the Extra Trees or **Extremely Randomized Trees** algorithm ensemble with k-fold cross validation on the Volatile Organic Compound GCMS readings. 

# In[1]:


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = KFold(n_splits=10, shuffle=True, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


# # Explore different amounts of trees

# In[2]:


import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot


# In[3]:


#get data
raw = pd.read_csv('data/cleaned/long.csv')
filled = raw.fillna(0)


# In[136]:


#store results from formula below
vary_trees = pd.DataFrame(columns=["mean","std"])


# In[145]:


# get a list of models to evaluate
def get_models():
	models = dict()
	# define number of trees to consider
	n_trees = [10, 50, 100, 150, 200, 250, 300, 350, 500, 750, 1000]
	for n in n_trees:
		models[str(n)] = ExtraTreesClassifier(n_estimators=n)
	return models
 
# define dataset
X, y = filled.iloc[:,3:], filled.iloc[:,1]
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names, times = list(), list(), list()
for name, model in models.items():
	# evaluate the model
	start = time.time()
	scores = evaluate_model(model, X, y)
	end = time.time()
	# store the results
	times.append(end-start)
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	s = pd.Series(data=[np.mean(scores), np.std(scores), np.mean(times)],name=name, index=['mean','std', 'time'])
	vary_trees = vary_trees.append(s)
	print('>%s %.3f (%.3f) (%.3f)' % (name, np.mean(scores), np.std(scores), np.mean(times)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.title('Varying The Amount of Trees')
pyplot.xlabel('Number of Trees')
pyplot.ylabel('Accuracy')
pyplot.savefig('images/ex_tree/vary_trees.png')
pyplot.show()


# In[146]:


#find average of all stored results
avg_vary_trees = pd.DataFrame(columns=['mean', 'std', 'time'])
for i in [10, 50, 100, 150, 200, 250, 300, 350, 500, 750, 1000]:
    i = str(i)
    s = pd.Series(vary_trees.T[i].mean(axis=1), name=i, index=['mean','std', 'time'])
    avg_vary_trees = avg_vary_trees.append(s)


# In[147]:


avg_vary_trees.sort_values(by='mean', ascending=0).head(7)


# In[148]:


avg_vary_trees.to_csv('data/cleaned/ex_trees/avg_vary_trees.csv', index=True)


# In[149]:


line1 = pyplot.plot(avg_vary_trees.index, avg_vary_trees['mean'], label='Mean')
line2 = pyplot.plot(avg_vary_trees.index, avg_vary_trees['std'], c='yellow', label='Std')
pyplot.title('Accuracy by Varying Trees')
pyplot.xlabel('Number of Trees')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.yticks(ticks=np.arange(0,1.1,0.1))
pyplot.savefig('images/ex_tree/vary_trees2.png')
pyplot.show()


# ## Explore different amounts of features

# In[150]:


#store results from formula below
vary_feat = pd.DataFrame(columns=["mean","std"])


# In[166]:


# get a list of models to evaluate
def get_models():
	models = dict()
	# explore number of features from 1 to 20
	depths = [i for i in range(1,67)] + [None]
	for i in range(1, 67):
		models[str(i)] = ExtraTreesClassifier(max_features=i)
	return models
 
# define dataset
X, y = filled.iloc[:,3:], filled.iloc[:,1]
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names, times = list(), list(), list()
for name, model in models.items():
	# evaluate the model
	start = time.time()
	scores = evaluate_model(model, X, y)
	end = time.time()
	# store the results
	times.append(end-start)
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	s = pd.Series(data=[np.mean(scores), np.std(scores), np.mean(times)],name=name, index=['mean','std', 'time'])
	vary_feat = vary_feat.append(s)
	print('>%s %.3f (%.3f) (%.3f)' % (name, np.mean(scores), np.std(scores), np.mean(times)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# In[167]:


#find average of all stored results
avg_vary_feat = pd.DataFrame(columns=['mean', 'std', 'time'])
for i in range(1, 67):
    i = str(i)
    s = pd.Series(vary_feat.T[i].mean(axis=1), name=i, index=['mean','std', 'time'])
    avg_vary_feat = avg_vary_feat.append(s)


# In[168]:


avg_vary_feat.sort_values(by='mean', ascending=0)


# In[169]:


avg_vary_feat.to_csv('data/cleaned/ex_trees/avg_vary_feat.csv', index=True)


# In[170]:


pyplot.figure(figsize=(15,7))
line1 = pyplot.plot(avg_vary_feat.index, avg_vary_feat['mean'], label='Mean')
line2 = pyplot.plot(avg_vary_feat.index, avg_vary_feat['std'], c='yellow', label='Std')
pyplot.title('Accuracy of Varying Random Subspace')
pyplot.xlabel('Number of Features')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.yticks(ticks=np.arange(0,1.1,0.1))
pyplot.xticks(ticks=np.arange(0,66,5))
pyplot.savefig('images/ex_tree/vary_features2.png')
pyplot.show()


# ## Exploring number of minimum splits

# In[157]:


min_trees = pd.DataFrame(columns=["mean","std"])


# In[206]:


# get a list of models to evaluate
def get_models():
	models = dict()
	# explore the number of samples per split from 2 to 14
	for i in range(2, 15):
		models[str(i)] = ExtraTreesClassifier(min_samples_split=i)
	return models

#get models to evaluate
models = get_models()
# evaluate the models and store results
results, names, times = list(), list(), list()
for name, model in models.items():
	# evaluate the model
	start = time.time()
	scores = evaluate_model(model, X, y)
	end = time.time()
	# store the results
	times.append(end-start)
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	s = pd.Series(data=[np.mean(scores), np.std(scores), np.mean(times)],name=name, index=['mean','std', 'time'])
	min_trees = min_trees.append(s)
	print('>%s %.3f (%.3f) (%.3f)' % (name, np.mean(scores), np.std(scores), np.mean(times)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.title('Minimum Number of Splits by Accuracy')
pyplot.show()


# In[207]:


#find average of all stored results
avg_min_trees = pd.DataFrame(columns=['mean', 'std', 'time'])
for i in range(2, 15):
    i = str(i)
    s = pd.Series(min_trees.T[i].mean(axis=1), name=i, index=['mean','std', 'time'])
    avg_min_trees = avg_min_trees.append(s)


# In[208]:


avg_min_trees.sort_values(by='mean', ascending=0)


# In[209]:


line1 = pyplot.plot(avg_min_trees.index, avg_min_trees['mean'], label='Mean')
line2 = pyplot.plot(avg_min_trees.index, avg_min_trees['std'], c='yellow', label='Std')
pyplot.title('Accuracy of Varying Number of Splits')
pyplot.xlabel('Number of Splits')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.yticks(ticks=np.arange(0,1.1,0.1))
pyplot.savefig('images/ex_tree/vary_splits2.png')
pyplot.show()


# In[177]:


avg_min_trees.to_csv('data/cleaned/ex_trees/avg_min_split.csv', index=True)


# ## Best Performing Model According to experimentation
# 1. amount of trees = 100
# 2. number of features = all (66) ??
# 3. number of minimum splits = 2
# 
# Experimentation has proved that for the most part, the default values of Extra Tree's hyperparameters are not the best suited values for this particular model. Through thorough experimentation, we deduce that the number of minimum splits is best suited to 7 in this particular case over time. Although it should be noted that the gain in variance and standard deviation is miniscule. As well as this, similarly to the random forest model, experimentation results suggest that the number of features to be included remains ambiguous. The general rule of thumb for the number of features to be used in a classification task is the sqrt(number of features), which in our case rounds to 8. However, including only 1 to 10 features on average achieves the lowest results for mean accuracy, although the std is marginally smaller than for other hyperparameter selections such as 50.

# In[210]:


final_results = pd.DataFrame(columns=['mean accuracy', 'mean std'])


# In[212]:


final_model = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', min_samples_split=2)

# define dataset
X, y = filled.iloc[:,3:], filled.iloc[:,1]

results, names = list(), list()
for i in range(50):
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    s = pd.Series(data=[np.mean(scores), np.std(scores)],name=name, index=['mean','std'])
    final_results = final_results.append(s)
    print('>%s %.3f (%.3f)' % (i, np.mean(scores), np.std(scores)))


# In[213]:


final_results.mean()


# In[ ]:




