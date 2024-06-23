#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_df = pd.read_csv("train.csv")
titanic_df.head()


# In[3]:


missing_values = titanic_df.isnull().sum()
print("Missing Values:")
print(missing_values)

titanic_df.fillna({'Age': titanic_df['Age'].median()}, inplace=True)

if 'Cabin' in titanic_df.columns:
    titanic_df.drop('Cabin', axis=1, inplace=True)
    print("Column 'Cabin' has been dropped.")
else:
    print("Column 'Cabin' does not exist in the DataFrame.")

print("\nAfter handling missing values:")
print(titanic_df.isnull().sum())


# In[4]:


print("\nSummary Statistics:")
print(titanic_df.describe())


# In[5]:


plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[6]:


gender_count = titanic_df['Sex'].value_counts()
survived_gender_count = titanic_df.groupby('Sex')['Survived'].sum()

survival_rate = (survived_gender_count / gender_count) * 100

plt.figure(figsize=(8, 6))
sns.barplot(x=survival_rate.index, y=survival_rate.values, hue=survival_rate.index, palette='coolwarm', dodge=False)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate (%)')
plt.ylim(0, 100)
plt.show()


# In[7]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Pclass')
plt.show()     


# In[8]:


survived = titanic_df[titanic_df['Survived'] == 1]
not_survived = titanic_df[titanic_df['Survived'] == 0]

plt.figure(figsize=(10, 6))
sns.histplot(x='Age', data=survived, kde=True, color='blue', label='Survived', alpha=0.7)
sns.histplot(x='Age', data=not_survived, kde=True, color='red', label='Did not survive', alpha=0.7)
plt.title('Survival Count by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()


# In[9]:


numeric_columns = titanic_df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()

corr_order = correlation_matrix.abs().sum().sort_values(ascending=False).index
correlation_matrix = correlation_matrix.reindex(index=corr_order, columns=corr_order)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5, center=0)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




