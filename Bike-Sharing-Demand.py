#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


bike_data = pd.read_csv('day.csv')


# In[3]:


print(bike_data.head())


# In[4]:


print(bike_data.info())


# In[5]:


print(bike_data.describe())


# In[6]:


plt.figure(figsize=(10, 6))
sns.histplot(bike_data['cnt'], bins=30, kde=True)
plt.title('Distribution of Total Bike Rentals (cnt)')
plt.xlabel('Total Bike Rentals (cnt)')
plt.ylabel('Frequency')
plt.show()


# In[7]:


numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=feature, y='cnt', data=bike_data)
    plt.title(f'{feature} vs Total Bike Rentals (cnt)')
plt.tight_layout()
plt.show()


# In[8]:


categorical_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=feature, y='cnt', data=bike_data)
    plt.title(f'{feature} vs Total Bike Rentals (cnt)')
plt.tight_layout()
plt.show()


# In[9]:


season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
weathersit_mapping = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}


# In[10]:


bike_data['season'] = bike_data['season'].map(season_mapping)
bike_data['weathersit'] = bike_data['weathersit'].map(weathersit_mapping)


# In[11]:


bike_data = bike_data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)


# In[12]:


print(bike_data.head())


# In[13]:


bike_data_encoded = pd.get_dummies(bike_data, columns=['season', 'weathersit'], drop_first=True)


# In[14]:


X = bike_data_encoded.drop('cnt', axis=1)
y = bike_data_encoded['cnt']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[17]:


y_train_pred = model.predict(X_train)


# In[18]:


residuals = y_train - y_train_pred


# In[19]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[21]:


y_test_pred = model.predict(X_test)


# In[22]:


r2 = r2_score(y_test, y_test_pred)
print(f'R-squared score: {r2}')


# In[23]:


print("Intercept:", model.intercept_)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


# In[ ]:




