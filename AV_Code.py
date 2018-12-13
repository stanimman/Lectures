
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;


# In[59]:


train = pd.read_csv('Downloads\\Train_UWu5bXk.csv')
test = pd.read_csv('Downloads\\Test_u94Q5KV.csv')


# In[13]:


print(train.shape)
print(test.shape)


# In[4]:


Train_file.head()


# In[9]:


all_data = pd.concat([(train.drop(['Item_Outlet_Sales'],axis=1)),test])


# In[32]:


all_data.isnull().sum()


# In[23]:


all_data[all_data.Item_Weight.isnull()]


# In[26]:


all_data[all_data.Item_Identifier =='FDP10']


# In[ ]:


all_data.groupby(['Item_Identifier','Item_Weight'])['Item_Weight'].mean()


# In[31]:


all_data['Imputed_Weight'] = all_data.groupby('Item_Identifier')['Item_Weight'].transform('mean') # Woow Tranform has been missed till now Really a killer function


# In[27]:


all_data.Outlet_Size.value_counts()


# In[29]:


all_data.loc[all_data.Outlet_Size.isnull(),'Outlet_Size'] = 'Small'


# In[33]:


Tier_Mapping ={'Tier 3':1,'Tier 2':2,'Tier 1':3}
Fat_Mapping = {'Low Fat':'Low Fat', 'Regular':'Regular', 'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}


# In[34]:


all_data['Mapped_Tier'] = all_data['Outlet_Location_Type'].map(Tier_Mapping)
all_data['Mapped_Fat'] = all_data['Item_Fat_Content'].map(Fat_Mapping)


# In[35]:


all_data = pd.get_dummies(data=all_data, columns=['Item_Type', 'Outlet_Type','Outlet_Size','Mapped_Fat'])


# In[36]:


#Remove Unwanted Catagorical Variable from the Dataframe
all_data = all_data.drop(['Item_Weight','Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Item_Fat_Content'], axis=1)


# In[38]:


all_data.shape


# In[39]:


# Cleaning and readying data for scikit learn operation
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    print(len(indices_to_keep))
    return df[indices_to_keep].astype(np.float64)


# In[40]:


all_data = clean_dataset(all_data)


# In[41]:


# Normalization
from sklearn.preprocessing import MinMaxScaler
stdsc = StandardScaler()
all_data = mms.fit_transform(all_data)


# In[ ]:


all_data


# In[44]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]


# In[46]:


# In[320]:

y = train['Item_Outlet_Sales']/train['Item_MRP']


# In[47]:


print(X_train.shape)
print(X_test.shape)
print(y.shape)


# In[48]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
get_ipython().magic(u'matplotlib')
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()


# In[49]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[54]:


model_ridge = Ridge()
alphas = [0.005,0.05, 0.1, 0.3,0.5,1]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[55]:


# Lasso Model
model_lasso = LassoCV().fit(X_train, y)
rmse_cv(model_lasso).mean()


# In[56]:


preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds = model_lasso.predict(X_test)


# In[57]:


#Ridge Solution RidgeCV is Ridge regression with built in cross Validation , no need to find alpha seperatly
model_ridge = RidgeCV().fit(X_train, y)
rmse_cv(model_ridge).mean()
preds = model_ridge.predict(X_test)


# In[60]:


preds = test['Item_MRP']*preds


# In[62]:


solution = pd.DataFrame({"Item_Identifier":test.Item_Identifier,"Outlet_Identifier":test.Outlet_Identifier,"Item_Outlet_Sales":preds})
solution.to_csv("ridge_sol.csv",index= False, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])

