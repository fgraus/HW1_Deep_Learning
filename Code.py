#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


trainingList = pd.DataFrame(columns = ['label'])
evaluationList = pd.DataFrame(columns = ['label'])
validateList = pd.DataFrame(columns = ['label'])

for root, dirs, files in os.walk('Food-5K/training', topdown=False):
    for file in files:
        try:
            image = Image.open(root+'/'+file)
            imageResize = tf.image.resize(image, [50,50], method='nearest', preserve_aspect_ratio=False)
            imageFlatten = tf.reshape(imageResize, [7500]).numpy()
            trainingList = pd.concat([trainingList, pd.DataFrame(imageFlatten).T], ignore_index=True)
            trainingList.loc[trainingList.shape[0]-1,'label'] = int(file[0])
        except Exception:
            print('Error getting training image on image ' + file)
            
    for root, dirs, files in os.walk('Food-5K/evaluation', topdown=False):
        for file in files:
            try:
                image = Image.open(root+'/'+file)
                imageResize = tf.image.resize(image, [50,50], method='nearest', preserve_aspect_ratio=False)
                imageFlatten = tf.reshape(imageResize, [7500]).numpy()
                evaluationList = pd.concat([evaluationList, pd.DataFrame(imageFlatten).T], ignore_index=True)
                evaluationList.loc[evaluationList.shape[0]-1,'label'] = int(file[0])
            except Exception:
                print('Error getting evaluation image on image ' + file)
            
    for root, dirs, files in os.walk('Food-5K/validation', topdown=False):
        for file in files:
            try:
                image = Image.open(root+'/'+file)
                imageResize = tf.image.resize(image, [50,50], method='nearest', preserve_aspect_ratio=False)
                imageFlatten = tf.reshape(imageResize, [7500]).numpy()
                validateList = pd.concat([validateList, pd.DataFrame(imageFlatten).T], ignore_index=True)
                validateList.loc[validateList.shape[0]-1,'label'] = int(file[0])
            except Exception:
                print('Error getting testing image on image ' + file)
                
trainPlusValidate = pd.concat([trainingList, validateList], ignore_index=True)
trainPlusValidate = trainPlusValidate.sample(frac=1)


# In[3]:


get_ipython().run_line_magic('store', 'trainingList')
get_ipython().run_line_magic('store', 'evaluationList')
get_ipython().run_line_magic('store', 'validateList')


# In[2]:


get_ipython().run_line_magic('store', '-r trainingList')
get_ipython().run_line_magic('store', '-r evaluationList')
get_ipython().run_line_magic('store', '-r validateList')


# In[14]:


plt.hist(trainPlusValidate['label'])


# In[4]:


gbm_param_grid = {
    'n_neighbors' : [10, 20, 30, 40, 50, 80, 100, 300, 600, 1000],
    'weights': ['uniform'],
    'metric': ['manhattan']
}

gs = RandomizedSearchCV(
    KNeighborsClassifier(),
    gbm_param_grid,
    cv = 3,
    verbose = 4,
    n_jobs = -1
    )

gs.fit(trainPlusValidate.drop(['label'], axis=1), trainPlusValidate['label'].astype('int'))


# In[5]:


get_ipython().run_line_magic('store', 'gs')


# In[3]:


get_ipython().run_line_magic('store', '-r gs')


# In[16]:


nnTrainScore = gs.score(trainPlusValidate.drop(['label'], axis=1), trainPlusValidate['label'].astype('int'))
nnEvaluateScore = gs.score(evaluationList.drop(['label'], axis=1), evaluationList['label'].astype('int'))

print('Train score ' + str(nnTrainScore))
print('Evaluate score ' + str(nnEvaluateScore))


# In[45]:


gs.best_params_


# In[30]:


params_nn = pd.concat([pd.DataFrame(gs.cv_results_["params"]),pd.DataFrame(gs.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
params_nn = params_nn.sort_values(by=['n_neighbors'])
params_nn = params_nn.drop(['weights', 'metric'], axis=1)


# In[31]:


params_nn


# In[33]:


params_nn.plot(kind='line',x='n_neighbors',y='Accuracy')


# In[12]:


gs_svm.best_params_


# In[13]:


svm_param_grid = {
    'learning_rate': ['constant'],
    'eta0': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [300, 600, 900, 1200, 1500],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]
}

gs_svm = GridSearchCV(SGDClassifier(loss='hinge'),
            svm_param_grid,
            cv = 3,
            verbose = 4,
            n_jobs = -1
            )

gs_svm.fit(trainPlusValidate.drop(['label'], axis=1), trainPlusValidate['label'].astype('int'))


# In[ ]:


get_ipython().run_line_magic('store', 'gs_svm')


# In[ ]:


get_ipython().run_line_magic('store', '-r gs_svm')


# In[17]:


svmTrainScore = gs_svm.score(trainPlusValidate.drop(['label'], axis=1), trainPlusValidate['label'].astype('int'))
svmEvaluateScore = gs_svm.score(evaluationList.drop(['label'], axis=1), evaluationList['label'].astype('int'))

print('Train score ' + str(svmTrainScore))
print('Evaluate score ' + str(svmEvaluateScore))


# In[34]:


params_svm = pd.concat([pd.DataFrame(gs_svm.cv_results_["params"]),pd.DataFrame(gs_svm.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[44]:


params_svm.loc[params_svm['max_iter'] == 1200 ].sort_values(by=['Accuracy']).drop(['learning_rate'], axis=1)


# In[20]:


# Print a bar chart with groups

import numpy as np
import matplotlib.pyplot as plt

# set height of bar
# length of these lists determine the number
# of groups (they must all be the same length)
bars1 = [nnTrainScore, svmTrainScore]
bars2 = [nnEvaluateScore, svmEvaluateScore]

# set width of bar. To work and supply some padding
# the number of groups times barWidth must be
# a little less than 1 (since the next group
# will start at 1, then 2, etc).

barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='training')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='evaluation')

# Add xticks on the middle of the group bars
plt.xlabel('Accuracy', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Nearest Neighbors', 'SVM'])

# Create legend & Show graphic
plt.legend()
plt.show()
#plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf

