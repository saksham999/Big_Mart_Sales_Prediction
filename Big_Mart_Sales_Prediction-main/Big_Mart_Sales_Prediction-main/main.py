#!/usr/bin/env python
# coding: utf-8

# In[338]:


import pandas as pd


# In[339]:


df=pd.read_csv("train_v9rqX0R.csv")
df.head(5)


# In[340]:


df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df=df.drop(['Item_Identifier','Outlet_Identifier','New_Item_Type'],axis=1)
df.head(5)


# In[341]:


for i in df:
    if df.dtypes[i]=='object':
        print(i)
        print(df[i].value_counts())


# In[342]:


df['Item_Visibility']=df.loc[:,'Item_Visibility'].replace([0],[df['Item_Visibility'].mean()])


# In[343]:


df['Item_Fat_Content']=df['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})


# In[344]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Item_Fat_Content']=labelencoder.fit_transform(df['Item_Fat_Content'])
df['Item_Type']=labelencoder.fit_transform(df['Item_Type'])
df['Outlet_Location_Type']=labelencoder.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type']=labelencoder.fit_transform(df['Outlet_Type'])
df.head(5)


# In[345]:


df=df.fillna({'Item_Weight':df['Item_Weight'].mean()})
df=df.fillna({'Outlet_Size':df['Outlet_Size'].mode()[0]})
df.head(5)


# In[346]:


df['Outlet_Size']=labelencoder.fit_transform(df['Outlet_Size'])
df.head(5)


# In[347]:


df.info()


# In[348]:


df['Outlet_Establishment_Year']=2013-df['Outlet_Establishment_Year']


# In[349]:


import seaborn as sns


# In[350]:


sns.displot(df['Item_Weight'])


# In[351]:


sns.displot(df['Item_Visibility'])


# In[352]:


sns.displot(df['Item_MRP'])


# In[353]:


sns.displot(df['Item_Outlet_Sales'])


# In[354]:


import numpy as np
df['Item_Outlet_Sales']=np.log(1+df['Item_Outlet_Sales'])


# In[355]:


sns.displot(df['Item_Outlet_Sales'])


# In[356]:


X=df.drop(['Item_Outlet_Sales'],axis=1)
y=df['Item_Outlet_Sales']


# In[357]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train the model
    model.fit(X, y)
    
    # predict the training set
    pred = model.predict(X)
    
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:", cv_score)


# In[358]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[359]:


df.head(5)


# In[360]:


model = Ridge(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
print(model.score(x,y))


# In[361]:


model = Lasso()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# In[362]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# In[363]:


model = RandomForestRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")


# In[364]:


x_test=pd.read_csv('test_AbJTz2l.csv')


# In[365]:


x_test.isna().sum()


# In[366]:


x_ans=pd.DataFrame(x_test['Item_Identifier'])
x_ans.insert(1,'Outlet_Identifier',x_test['Outlet_Identifier'],True)
x_test['New_Item_Type'] = x_test['Item_Identifier'].apply(lambda x: x[:2])
x_test['New_Item_Type'] = x_test['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
x_test.loc[x_test['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
x_test=x_test.drop(['Item_Identifier','Outlet_Identifier','New_Item_Type'],axis=1)
x_test['Item_Visibility']=x_test.loc[:,'Item_Visibility'].replace([0],[x_test['Item_Visibility'].mean()])
x_test['Item_Fat_Content']=x_test['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
x_test['Item_Weight']=x_test['Item_Weight'].fillna(x_test['Item_Weight'].mean())
x_test['Outlet_Size']=x_test['Outlet_Size'].fillna(x_test['Outlet_Size'].mode()[0])
x_test['Item_Fat_Content']=labelencoder.fit_transform(x_test['Item_Fat_Content'])
x_test['Item_Type']=labelencoder.fit_transform(x_test['Item_Type'])
x_test['Outlet_Location_Type']=labelencoder.fit_transform(x_test['Outlet_Location_Type'])
x_test['Outlet_Type']=labelencoder.fit_transform(x_test['Outlet_Type'])
x_test['Outlet_Size']=labelencoder.fit_transform(x_test['Outlet_Size'])


# In[367]:


x_test.info()


# In[368]:


y_test=model.predict(x_test)


# In[369]:


y_test=np.exp(y_test)+1


# In[370]:


x_ans.insert(2,"Item_Outlet_Sales",y_test,True)


# In[371]:


x_ans.head(5)


# In[372]:


x_ans.to_csv("sample_submission_8RXa3c6.csv",index=False)


# In[ ]:





# In[ ]:




