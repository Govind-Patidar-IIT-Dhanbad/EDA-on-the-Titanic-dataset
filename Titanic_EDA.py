#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# EDA ON TITATIC DATASET FOR SURVIVOR PREDICTION

# 
# #### Data Preprocessing
#  
#  1.Load data
#  2.Data Overview
#  
# #### Data Cleaning
# 
#  1.Handling Null values
#  2.Handling Outliers
#  
# #### Data Analysis
# 
#  1.Univariate Analysis
#  2.Multivariate Analysis
# 
# #### Feature Engineering
# 
#  1.Creating new columns
#  2.Modifying columns

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df2 = pd.read_csv('D:\\EDA project1\\Titanic.csv')
df2


# In[3]:


# Get the shape of the DataFrame
df2.shape
#df2.shape[0] = 1310  # df2.shape[1]= 14


# In[4]:


# Display the first two rows of the DataFrame df2
df2.head(2)


# In[5]:


# Display the last 3 rows of the DataFrame df2
df2.tail(3)


# In[6]:


# to get information about the dataframe
df2.info()


# In[7]:


# descriptive statistics of the DataFrame
df2.describe()


# In[8]:


# Check the number of null values in df2
df2.isnull().sum()


# #### Treating null values...> Rules:
# 1.null values greater than 40% --> the column will be dropped
# 2.null values will be less than 5%--> we can drop the null values 
# 3.null values will be between 5 to 40-->#replace the values with either mean / median (depending on presence of outliers)

# In[9]:


#Get the column names of a DataFrame
df2.columns


# In[10]:


n = df2.shape[0]

drop_clm = []
drop_na = []
fill_na = []

# Loop through each column in df2
for i in df2.columns:
    x = df2[i].isnull().sum()
    
    # Check if the percentage of null values is greater than or equal to 40%
    if (x / n * 100) >= 40:
        drop_clm.append(i)
    
    # Check if the percentage of null values is less than or equal to 5%
    elif (x / n * 100) <= 5:
        drop_na.append(i)
    
    # Otherwise, add the column to fill_na list
    else:
        fill_na.append(i)

drop_clm




# In[11]:


fill_na


# In[12]:


drop_na


# In[13]:


# Remove drop_clm list columns from df2
df2.drop(drop_clm,axis=1, inplace = True)


# In[14]:


df2.columns


# In[15]:


# Again calculate the percentage of missing values in each column of df2
n=df2.shape[0]
for i in df2.columns:
    x=df2[i].isnull().sum()
    print(i,(x/n)*100)


# In[16]:


#filling missing values

df2["age"]=df2['age'].fillna(df2.age.median())
df2.age.isnull().sum()


# In[17]:


df2["embarked"] = df2.embarked.fillna(df2.embarked.mode)
df2.embarked.isnull().sum()


# In[18]:


df2.dropna(inplace=True)
df2.isnull().sum()


# In[19]:


# Display a boxplot 
df2.boxplot()


# Box plot is showing column “age” and “fare” is having outliers.
# 

# Treating outliers:
# 
# #whether the data type of column is numerical or object.
# #age is having outliers ,,fill the missing values with median else mean 
# #embarked is categorical fill the missing values with mode
# 
# 

# In[20]:


# Calculate the interquartile range (IQR)
q1 = df2['age'].quantile(0.25)
q2 = df2['age'].median()
q3 = df2['age'].quantile(0.75)
iqr = q3 - q1

# Print the quartile values and IQR
q1, q2, q3, iqr


# In[21]:


# Calculate the lower and upper fencing for outliers
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

lower,upper


# In[22]:


df2.loc[(df2.age >= upper) | (df2.age <= lower), 'age']


# In[23]:


df2.age[21]


# In[24]:


df2.age[df2.age > upper]


# assignment operation is performed directly on the DataFrame column. In pandas, when we perform an assignment like:                  df.loc[condition, 'column'] = value, it modifies the DataFrame in place by default.

# In[25]:


#Remove outliers and replace them with NaN

age_outliers = (df2['age'] >= upper) | (df2['age'] < lower)
df2.loc[age_outliers, 'age'] = np.nan

#df2['age'] = np.where((df2['age'] < lower) | (df2['age'] > upper), np.nan, df2['age'])


# In[26]:


df2.age.dtype


# In[27]:


df2['age'] >= upper


# In[28]:


#Check for null values
df2.isnull().sum()


# In[59]:


df2['age'] = df2.age.fillna(df2.age.median())


# In[58]:


df2.isnull().sum()


# #df['column'].replace(to_replace='old_value', value='new_value')
# #Do not use inplace  = true. otherwise it will give "None" for each value
# 

# #if you attemt some mistake by inplace = true
# #try this: df2['sex'] = df1['sex']

# #df1 = pd.read_csv('D:\\EDA project1\\Titanic.csv')

# In[32]:


import pandas as pd

#Calculate the interquartile range (IQR)
q1 = df2['age'].quantile(0.25)
q2 = df2['age'].median()
q3 = df2['age'].quantile(0.75)
iqr = q3 - q1

# Print the quartile values and IQR
q1, q2, q3, iqr


# In[33]:


lower = q1-iqr*1.5
upper = q1+iqr*1.5


# In[34]:


lower, upper


# In[35]:


df2.fare[df2.fare < lower]


# In[36]:


df2.fare[df2.fare > upper]


# In[37]:


df2.loc[(df2.fare >= upper) | (df2.fare < lower), 'fare'] = np.nan


# In[38]:


df2['fare'].isnull().sum()


# In[39]:


df2['fare'] = df2.fare.fillna(df2.fare.median())


# In[40]:


df2['fare'].isnull().sum()


# In[41]:


#Replace 'female' with 1 and 'male' with 0 in the 'sex' column of df2

df2['sex']=df2['sex'].replace('female',1)

df2['sex']=df2['sex'].replace('male',0)


# In[42]:


# Univariate Analysis

#Calculate total number of records and number of survived passengers

Total_counts = df2['survived'].shape[0]
Survived_count = df2['survived'].sum()
Death_count = Total_counts - Survived_count
Survived_prct = round(Survived_count/Total_counts*100,2)
Death_prct = round(Death_count/Total_counts*100,2)

print(Survived_count)
print(f"Among a total of {Total_counts} passengers, only {Survived_prct}% of them managed to survive.")

#Create a custom color palette for the countplot

cstm_plt = ['red', 'green']

sns.countplot(x='survived',hue = 'survived', data=df2, palette=cstm_plt)
plt.show() 


# In[43]:


# Customized the bins size and labels

bins = [0, 2, 4, 13, 20, 50, 70, 110]
labels = ['Infant', 'Toddler', 'Kid', 'Teen', 'Adult', 'Middle-aged', 'Old-aged']
df2['AgeGroup'] = pd.cut(df2['age'], bins=bins, labels=labels, right=False)


sns.countplot(x='AgeGroup', data=df2)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Passenger Count by Age Group')
plt.show()


# In[44]:


sns.displot(df2['age'])
plt.show()



# In[45]:


# Customized the bins size and labels

bins= [0,30,60,100,250,1000]
labels = ['Cheap','Average','Above Average','High','Expensive']
df2['Fare_Range'] = pd.cut(df2['fare'], bins=bins, labels=labels, right=False)

sns.countplot(x='Fare_Range', data=df2)
plt.xlabel('Fare_Range')
plt.ylabel('Count')
plt.title('Passenger Count by Fare Range')
plt.show()


# In[46]:


sns.countplot(x='pclass', data = df2)
plt.xlabel('Passanger class')
plt.ylabel('Count')
plt.title('Passenger Count by Class')
plt.show()
#1- First class
#2- Second class 
#3- Third class


# In[47]:


sns.countplot(x='sibsp', data = df2)
plt.xlabel('Number of siblings/spouses')
plt.title('Passenger Count by siblings/spouses ')
plt.show()


# In[48]:


sns.countplot(x='parch', data = df2)
plt.xlabel('Number of parents/children')
plt.title('Passenger Count by parents/children ')
plt.show()


# ### Univariate analysis conclusions:
# 
# 1.Among a total of 1308 passengers, only 38.23% of them managed to survive.
# 2.Mostly people were travelling on fare less than 3
# 3.Most people were travelling on 3rd class
# 4.Most people were from the age of 20 to 50
# 5.Very less passengers were travling with their siblings/spouses
# 6.Very less passengers were travling with their parents/children

# In[49]:


#Bivariate analysis

sns.countplot(hue='pclass',x='survived',data=df2)
plt.show()
pd.crosstab(df2['pclass'], df2['survived']).apply(lambda r:round((r/r.sum())*100, 1), axis=1)


# In[50]:


sns.countplot(hue='pclass',x='sex',data=df2)
plt.show()
pd.crosstab(df2['sex'], df2['survived']).apply(lambda r:round((r/r.sum())*100, 1), axis=1)


# In[51]:


# Calculate the male_passenger %

passenger_total = df2['sex'].shape[0]
male_total = 0
male_survived = 0
for i in df2[['sex','survived']].values:
    
    if i[0] == 0:    # Use nested if for two columns                 
        male_total+=1
        if i[1] == 1:
            male_survived +=1
            
male_pct = round(male_total/passenger_total*100)
srvd_pct = round(male_survived/male_total*100)

# Use f string for printing results

print(f"Among a total {passenger_total} passengers, {male_pct}% passengers were male and among them only {srvd_pct }% male were survived")
cstm_plt = ['red', 'green']

sns.countplot(hue='sex',x='survived',data=df2,palette=cstm_plt )
plt.show()
pd.crosstab(df2.sex,df2.survived).apply(lambda r:round((r/r.sum())*100, 1), axis=1)


# In[52]:


# Calculate the percentage of passengers, not having siblings

total = df2['sibsp'].shape[0]
count = 0
for i in df2['sibsp']:
    if i == 0:
        count+=1
       
Not_sibsp_pct = round((count/total*100),2)
print(f"Among a total {total}, {Not_sibsp_pct}% passengers were not having sibling or spouse")

sns.countplot(hue='sibsp',x='survived',data=df2)
plt.show()


# In[53]:


sns.countplot(hue='sex',x='pclass',data=df2)
plt.show()
pd.crosstab(df2['pclass'], df2['sex']).apply(lambda r: round((r/r.sum())*100,1), axis=1)


# #### Bivariate analysis
# 1.Most of the passengers were travelling in 3rd class; among them, only 25.6% could survive.
# 2.The max number of female were traveling in 1st class while The max number of male were traveling in 3rd class.
# 3.Among a total of 1308 passengers, 64% were male, and among them only 19% could survive.
# 4.Among a total 1308, 68.04% passengers were not having sibling or spouse 
# 

# In[65]:


#Feature Engineering

# Create a new column by the name of family which will be the sum of sibSp and parch cols

df2['family_size']=df2['parch'] + df2['sibsp']


# In[66]:


#Enginner a new feature by the name of family type

def family_type(number):
    if number==0:
        return "Alone"
    elif number>0 and number<=4:
        return "Medium"
    else:
        return "Large"


# In[67]:


#Modifying column
df2['family_type']=df2['family_size'].apply(family_type)


# In[68]:


df2['family_type']

