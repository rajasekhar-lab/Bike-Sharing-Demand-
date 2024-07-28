#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[4]:


bike_data = pd.read_csv("day.csv")


# In[5]:


print(bike_data.head())


# In[6]:


print(bike_data.describe())


# In[7]:


print(bike_data.info())


# In[8]:


bike_data = bike_data.drop(['casual', 'registered', 'dteday'], axis=1)


# In[9]:


season_dict = {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}
weathersit_dict = {1: 'clear', 2: 'mist', 3: 'light_rain', 4: 'heavy_rain'}
bike_data['season'] = bike_data['season'].map(season_dict)
bike_data['weathersit'] = bike_data['weathersit'].map(weathersit_dict)


# In[10]:


bike_data = pd.get_dummies(bike_data, columns=['season', 'weathersit'], drop_first=True)


# In[11]:


print(bike_data.head())


# In[12]:


sns.histplot(bike_data['cnt'], kde=True)
plt.title('Distribution of Total Bike Rentals')
plt.xlabel('Total Bike Rentals')
plt.ylabel('Frequency')
plt.show()


# In[13]:


numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
for feature in numeric_features:
    sns.histplot(bike_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# In[14]:


for feature in numeric_features:
    sns.scatterplot(data=bike_data, x=feature, y='cnt')
    plt.title(f'{feature} vs Total Bike Rentals')
    plt.xlabel(feature)
    plt.ylabel('Total Bike Rentals')
    plt.show()


# In[15]:


categorical_features = ['yr', 'mnth', 'holiday', 'weekday', 'workingday']
for feature in categorical_features:
    sns.boxplot(data=bike_data, x=feature, y='cnt')
    plt.title(f'{feature} vs Total Bike Rentals')
    plt.xlabel(feature)
    plt.ylabel('Total Bike Rentals')
    plt.show()


# In[16]:


plt.figure(figsize=(12, 8))
correlation_matrix = bike_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[17]:


X = bike_data.drop(['cnt'], axis=1)
y = bike_data['cnt']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


print(X_train.isnull().sum())


# In[20]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


lm = LinearRegression()


# In[22]:


rfe = RFE(lm, n_features_to_select=10)


# In[23]:


rfe = rfe.fit(X_train_scaled, y_train)


# In[24]:


selected_features = X.columns[rfe.support_]
print('Selected features by RFE:', selected_features)


# In[25]:


X_train_sm = sm.add_constant(X_train_scaled)


# In[26]:


model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())


# In[27]:


vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm, i+1) for i in range(X_train_sm.shape[1]-1)]
print(vif)


# In[28]:


X_train_selected = X_train_scaled[:, rfe.support_]
X_test_selected = X_test_scaled[:, rfe.support_]


# In[29]:


X_train_sm_selected = sm.add_constant(X_train_selected)


# In[30]:


model_selected = sm.OLS(y_train, X_train_sm_selected).fit()
print(model_selected.summary())


# In[31]:


vif_selected = pd.DataFrame()
vif_selected['Features'] = selected_features
vif_selected['VIF'] = [variance_inflation_factor(X_train_sm_selected, i+1) for i in range(X_train_sm_selected.shape[1]-1)]
print(vif_selected)


# In[32]:


lm_final = LinearRegression()


# In[33]:


lm_final.fit(X_train_selected, y_train)


# In[34]:


y_pred = lm_final.predict(X_test_selected)


# In[35]:


r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)


# In[36]:


n = X_test_selected.shape[0]
p = X_test_selected.shape[1]
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
print('Adjusted R-squared score:', adjusted_r2)


# In[37]:


mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# In[38]:


residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[39]:


sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[42]:


plt.scatter(y_pred, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(0, color='r', linestyle='--')
plt.show()


# In[43]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Bike Rentals')
plt.show()


# In[ ]:




