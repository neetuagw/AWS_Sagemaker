
# coding: utf-8

# In[20]:


from sagemaker import get_execution_role

role = get_execution_role()
bucket = 'qbotsbucket'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import re
import io
import os
import time
import json
import sagemaker.amazon.common as smac
import sagemaker
from sagemaker.predictor import csv_serializer, json_deserializer

#Providing path of the File
input_prefix = 'Example-Datasets/50_Startups2.csv'

#Providing location of the file within bucket
data_location = 's3://{}/{}'.format(bucket, input_prefix)

# reading CSV file for train data
train_data = pd.read_csv(data_location)
train_X = train_data.iloc[:, :-1].values
train_y = train_data.iloc[:, -1].values

#Plot data on graph
plt.plot(train_data)
plt.show()

#Labeling dataset as required for LinearLearner Algorithm
"""array_data = np.array(train_data).astype('float32')
labels = array_data[:,1]"""

#Providing path of the output results
output_prefix = 'Example-Datasets/Output/'
#Setting output location in S3 bucket to upload output results
output_location = 's3://{}/{}'.format(bucket, output_prefix)


#Convert to RecordIO format as required by Amazon Sagemaker
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, np.array(train_X).astype('float32'), np.array(train_y).astype('float32'))
#smac.write_numpy_to_dense_tensor(buf, array_data, labels)
buf.seek(0)

#Uploading linear_traindata to S3
key = 'Example-Datasets/lineartrain2.data'
boto3.resource('s3').Bucket(bucket).Object(key).upload_fileobj(buf)
s3_train_data = 's3://{}/{}'.format(bucket, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[13]:


sess = sagemaker.Session()

#Setting Docker Image to deploy the model
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.m4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)


# In[14]:


#Providing hyperparameters
linear.set_hyperparameters(feature_dim=3,
                           predictor_type= 'regressor', 
                           mini_batch_size=10,
                           epochs=10,
                           num_models=2,
                           loss='absolute_loss',
                           use_bias = 'true')

#Fitting the model
linear.fit({'train': s3_train_data})

#Deploying the Model
linear_predictor = linear.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')


# Predictions

# In[16]:


test_prefix = 'Example-Datasets/50_Startups_test.csv'

#Providing location of the file within bucket
test_data_location = 's3://{}/{}'.format(bucket, test_prefix)

# reading CSV file for train and test data
test_data = pd.read_csv(test_data_location)

#Splitting the Test Data in test_X and test_Y
test_X = test_data.iloc[:, :-1].values
test_y = test_data.iloc[:, -1].values
print("Independent Variable of Test Set \n")
print(test_X, "\n")
print("Dependent Variable of Test Set \n")
print(test_y)

from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer


#Predicting test set results
results = linear_predictor.predict(test_X)
print(results)


# One Step Ahead Forecast

# In[19]:


one_step = np.array([r['score'] for r in results['predictions']])

print('One-step-ahead MdAPE = ', np.median(np.abs(test_y - one_step) / test_y))
plt.plot(np.array(test_y), label='actual')
plt.plot(one_step, label='forecast')
plt.legend()
plt.show()


# Clean Up

# In[21]:


sagemaker.Session().delete_endpoint(linear_predictor.endpoint)

