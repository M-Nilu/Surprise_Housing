#!/usr/bin/env python
# coding: utf-8

# # The Assignment comprises of
# - Data understanding and exploration
# - Data cleaning
# - Data preparation
# - Model building and evaluation
# - Observation and inference

# In[1]:


# importing all the important
import numpy as np
import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# reading the dataset
data = pd.read_csv("Housing_Details.csv", encoding = 'utf-8')
data.head()


# In[3]:


# Check the dimensions
data.shape


# In[4]:


# Check for column details
data.info()


# In[5]:


# To get the description of the dataset
data.describe()


# In[6]:


#checking duplicates
sum(data.duplicated(subset = 'Id')) == 0


# In[7]:


# Checking for percentage nulls
round(100*(data.isnull().sum()/len(data.index)), 2)


# # Outlier Check

# In[8]:


#Checking for outlier in the numerical columns
data.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# # Method to remove outliers.

# In[9]:


def remove_outliers(x,y):
    q1 = x[y].quantile(0.25)
    q3 = x[y].quantile(0.75)
    value = q3-q1
    lower_value  = q1-1.5*value
    higer_value = q3+1.5*value
    out= x[(x[y]<higer_value) & (x[y]>lower_value)]
    return out


# In[10]:


#Checking the shape of the dataframe
data.shape


# In[11]:


# since, it is clear that there are multiple columns with high nulls, lets group them together
data.columns[data.isnull().any()] 

null = data.isnull().sum()/len(data)*100
null = null[null>0]
null.sort_values(inplace=True, ascending=False)
null


# # we will first impute the categorical variables with 'None'

# In[12]:


# According to the data dictionary provided, the nulls in these columns indicates the absence of facility which may affect the price
# Hence, we will first impute the categorical variables with 'None'
null_with_meaning = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in null_with_meaning:
    data[i].fillna("none", inplace=True)


# In[13]:


# Check nulls once again

data.columns[data.isnull().any()] 

null_2 = data.isnull().sum()/len(data)*100
null_2 = null_2[null_2>0]
null_2.sort_values(inplace=True, ascending=False)
null_2


# In[14]:


# Will check these columns one by one
data['LotFrontage'].describe()


# In[15]:


data['GarageYrBlt'].describe()


# In[16]:


data['MasVnrArea'].describe()


# In[17]:


data['Electrical'].describe()


# In[18]:


# As per the data dictionary "LotFrontage" is Linear feet of street connected to property.  
# Since it is a numeric with a fair distribution, it can be imputed with similar 'Neighborhood' values

data['LotFrontage'] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
data["GarageYrBlt"].fillna(data["GarageYrBlt"].median(), inplace=True)
data["MasVnrArea"].fillna(data["MasVnrArea"].median(), inplace=True)
data["Electrical"].dropna(inplace=True)


# In[19]:


# Crosscheck the updated 'LotFrontage' column
data['LotFrontage'].describe()


# In[20]:


data['GarageYrBlt'].describe()


# In[21]:


data['MasVnrArea'].describe()


# In[22]:


data['Electrical'].describe()


# In[23]:


# Check the no. of rows retained
len(data.index)
len(data.index)/1460


# # Some EDA on the cleaned data
# All numeric (float and int) variables in the dataset

# In[25]:


data_numeric = data.select_dtypes(include=['float64', 'int64'])
data_numeric.head()


# #
# Target variable 'sale Price' vs a few select columns

# In[26]:


# plot 'Sale Price' with respect to 'Neighborhood'

plt.figure(figsize=(20, 8))
sns.barplot(x="Neighborhood", y="SalePrice", data= data)
plt.title("Sales Price with respect to Neighbourhood")
plt.xticks(rotation=90)


# #
# Properties in some of the Neighborhoods are high priced

# In[27]:


# plot 'overall condition' with respect to 'Saleprice'

plt.figure(figsize=(20, 8))
sns.barplot(x="OverallCond", y="SalePrice", data= data)
plt.title("Sales Price with respect to Overall Condition")
plt.xticks(rotation=90)


# In[28]:


# plot 'overall quality' with respect to 'Saleprice'

plt.figure(figsize=(20, 8))
sns.barplot(x="OverallQual", y="SalePrice", data= data)
plt.title("Sales Price with respect to Overall Quality")
plt.xticks(rotation=90)


# #
# Increase in the overall quality has a direct positive effect on the sale price

# In[29]:


sns.distplot(data['SalePrice'])


# In[30]:


data_raw = data.copy


# #
# Since the Saleprice figures are skewed towards left, we will apply the log transformation to obtain a centralized data

# In[31]:


#Log Transformation
data['SalePrice']=np.log1p(data['SalePrice'])


# In[32]:


sns.distplot(data['SalePrice'])


# In[33]:


# correlation matrix
cor = data_numeric.corr()
cor


# In[34]:


# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# #
# We can see that some of the variables are correlated
# 
# Before dropping these columns, we will first check their predictive power

# In[ ]:


# Checking the same with a pairplot 
sns.set()
cols = ['SalePrice', 'GrLivArea', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'TotRmsAbvGrd', 'GarageYrBlt']
sns.pairplot(data[cols], size = 2.5)
plt.show()


# #
# Drop columns that are correlated and not contributing to 'SalePrice'

# In[36]:


data = data.drop(['GarageCars'], axis = 1)
data = data.drop(['BsmtUnfSF'], axis = 1)
data = data.drop(['TotRmsAbvGrd'], axis = 1)
data = data.drop(['GarageYrBlt'], axis = 1)    

data.head()


# In[37]:


#Numeric columns
data.select_dtypes(exclude=['object'])


# In[38]:


# Analyse some important numeric columns
sns.jointplot(x='GrLivArea', y='SalePrice', data=data)
plt.show()


# In[39]:


# Removing some outliers on lower right side of 'GrLivArea'
data = remove_outliers(data,'GrLivArea')


# #
# Since the dataset is small it isn't advisable to do remove outliers.

# In[40]:


data.shape


# In[41]:


# Again plotting GeLivArea vs SalePrice
sns.jointplot(x = data['GrLivArea'], y = data['SalePrice'])
plt.show()


# In[42]:


# Lot frontage vs SalePrice 
sns.jointplot(x = data['LotFrontage'], y = data['SalePrice'])
plt.show()


# In[43]:


# LotArea vs SalePrice
sns.jointplot(x = data['LotArea'], y = data['SalePrice'])
plt.show()


# In[44]:


# 1stFlrSF vs SalePrice
sns.jointplot(x = data['1stFlrSF'], y = data['SalePrice'])
plt.show()


# In[45]:


# 2ndFlrSF vs SalePrice
sns.jointplot(x = data['2ndFlrSF'], y = data['SalePrice'])
plt.show()


# In[46]:


# OverallQual vs SalePrice
sns.jointplot(x = data['OverallQual'], y = data['SalePrice'])
plt.show()


# In[47]:


# OverallCond vs SalePrice
sns.jointplot(x=data['OverallCond'], y = data['SalePrice'])
plt.show()


# #
# Ground or First level houses i.e. '0' second floor Sq.Ft has also a steady increase

# # We can derive a column for 'Age of the property' when it was sold: Name it as 'PropAge'

# In[48]:


# PropAge -  Property Age from yearsold - yearbuilt
data['PropAge'] = (data['YrSold'] - data['YearBuilt'])
data.head()


# In[49]:


# PropAge vs SalePrice
sns.jointplot(x = data['PropAge'], y = data['SalePrice'])
plt.show()


# #
# Increase in Property Age shows a decreasing saleprice trend i.e newer the property, high is the value

# # Now we can drop the column Month sold and Year Sold, Year built and Year remodelled since it will not be required further

# In[50]:


data = data.drop(['MoSold'], axis = 1)
data = data.drop(['YrSold'], axis = 1)
data = data.drop(['YearBuilt'], axis = 1)
data = data.drop(['YearRemodAdd'], axis = 1)
data.head()


# In[51]:


data.Street.value_counts()


# In[52]:


data.Utilities.value_counts()


# In[53]:


# We can also drop columns that show very low variance and thus not required for predictions
data = data.drop(['Street'], axis = 1)
data = data.drop(['Utilities'], axis = 1)


# # Just to check the variance of these columns

# In[54]:


# l1 = ['Condition2', 'Heating', 'PoolQC', 'RoofMatl', 'BsmtCond', 'GarageQual', 'GarageCond', 'MiscVal', '3SsnPorch', 'FireplaceQu', 'BsmtHalfBath', 'BsmtFinSF2', 'Alley', 'MiscFeature', 'Fence', 'Functional']
l2= data.select_dtypes(include=['float64', 'int64'])
l2


# In[ ]:


for i in l2:
    print(data[i].value_counts())


# In[56]:


data = data.drop(['PoolQC','MiscVal', 'Alley', 'RoofMatl', 'Condition2', 'Heating', 'GarageCond', 'Fence', 'Functional' ], axis = 1)


# #
# These Columns were having high null values, some of which were imputed. After imputing, it was found that there was very little variance in the data. So we have decided to drop these columns.

# In[57]:


data.shape


# # 3. Data Preparation

# #
# Data Preparation
# 
# Let's now prepare the data and build the model.

# In[58]:


# Drop 'Id' from Dataframe

data = data.drop(['Id'], axis=1)
data.head()


# In[59]:


#type of each feature in data: int, float, object
types = data.dtypes
#numerical values are either type int or float
numeric_type = types[(types == 'int64') | (types == float)] 
#categorical values are type object
categorical_type = types[types == object]


# In[60]:


pd.DataFrame(types).reset_index().set_index(0).reset_index()[0].value_counts()


# In[61]:


#we should convert numeric_type to a list to make it easier to work with
numerical_columns = list(numeric_type.index)
print(numerical_columns)


# In[62]:


#Categorical columns
categorical_columns = list(categorical_type.index)
print(categorical_columns)


# #
# Creating Dummy columns to convert categorical into numerical

# In[63]:


data = pd.get_dummies(data, drop_first=True )
data.head()


# In[64]:


X = data.drop(['SalePrice'], axis=1)

X.head()


# In[65]:


# Putting response variable to y
y = data['SalePrice']

y.head()


# In[66]:


# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=50)


# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


scaler = StandardScaler()

X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']] = scaler.fit_transform(X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']])

X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']] = scaler.fit_transform(X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']])


# In[69]:


X_train.head()


# In[71]:


X_test.head()


# # 4. Model Building and Evaluation

# #
# Lets first check the model using Linear Regression and RFE (OPTIONAL)

# In[72]:


# Importing RFE and LinearRegression
# Since there are more variables to be analysed, we will used the automated feature elimination process (RFE)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[74]:


# Running RFE 
# Since there are more than 250 variables for analysis, we will run RFE to select some that have high predictive power
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[76]:


# running RFE for top 100 variables
rfe = RFE(estimator=lm, n_features_to_select=100)            
rfe.fit(X_train, y_train)


# In[77]:


# Check the ranks
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[78]:


# Select the top 100 variables

col = X_train.columns[rfe.support_]
col


# In[79]:


X_train.columns[~rfe.support_]


# In[80]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[81]:


X_train_rfe = pd.DataFrame(X_train[col])


# In[82]:


X_train_rfe.head()


# In[83]:


X_train_rfe.shape


# In[84]:


# predict
y_train_pred = lm.predict(X_train)
metrics.r2_score(y_true=y_train, y_pred=y_train_pred)


# In[85]:


y_test_pred = lm.predict(X_test)
metrics.r2_score(y_true=y_test, y_pred=y_test_pred)


# #
# Since the Test R2 is too low, we will check for some alternate methods of Regression

# In[86]:


# Check the ranks
list(zip(X_test.columns,rfe.support_,rfe.ranking_))


# In[87]:


# Select the top 100 variables

col1 = X_test.columns[rfe.support_]
col1


# In[88]:


X_test_rfe = X_test[col1]


# In[89]:


X_test_rfe.head()


# # Lasso and Ridge Regression

# #
# let's now try predicting house prices and perform lasso and ridge regression.

# #
# Lasso Regression

# In[90]:


# Checking the dimension of X_train & y_train
print("X_train", X_train.shape)
print("y_train", y_train.shape)


# In[91]:


# Applying Lasso

# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[93]:


# cv_results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=1]
cv_results.head()


# In[94]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[95]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# #
# From the above graph we can see that the Negative Mean Absolute Error is quite low at alpha = 0.4 and stabilises thereafter,
# but we will choose a low value of alpha to balance the trade-off between Bias-Variance
# and to get the coefficients of smallest of features.

# In[96]:


# At alpha = 0.01, even the smallest of negative coefficients that have some predictive power towards 'SalePrice' have been generated

alpha = 0.01
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
lasso.coef_


# #
# The advantage of this technique is clearly visible here as Lasso brings the coefficients of insignificant features to zero

# In[97]:


# lasso model parameters
model_parameters = list(lasso.coef_ )
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[98]:


# lasso regression
lm = Lasso(alpha=0.01)
lm.fit(X_train, y_train)

# prediction on the test set(Using R2)
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))


# In[99]:


print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# #
#  The R2 values for Train and Test matches well, indicating an optimum model

# In[100]:


# Creating a dataframe for the coefficients obtained from Lasso
mod = list(zip(cols, model_parameters))


# In[101]:


para = pd.DataFrame(mod)
para.columns = ['Variable', 'Coeff']
para.head()


# In[102]:


# sort the coefficients in ascending order
para = para.sort_values((['Coeff']), axis = 0, ascending = False)
para


# In[103]:


# Chose variables whose coefficients are non-zero
pred = pd.DataFrame(para[(para['Coeff'] != 0)])
pred


# In[104]:


# These 16 variables obtained from Lasso Regression can be concluded to have the strong effect on the SalePrice
pred.shape


# In[105]:


Lassso_var = list(pred['Variable'])
print(Lassso_var)


# In[106]:


X_train_lasso = X_train[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]
                        
X_train_lasso.head()


# In[107]:


X_train_lasso.shape


# In[108]:


X_test_lasso = X_test[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]
                        
X_test_lasso.head()


# # Ridge Regression

# In[109]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 


# In[110]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=5]
cv_results.head()


# In[111]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# #
# since the Negative Mean Absolute Error stabilises at alpha = 2, we will choose this for further analysis

# In[112]:


alpha = 2
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_


# In[113]:


# ridge model parameters
model_parameters = list(ridge.coef_)
model_parameters.insert(0, ridge.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[114]:


# ridge regression
lm = Ridge(alpha=2)
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))


# In[115]:


print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# #
# It is visible that the model performance is better than Lasso. The train and the test scores are matching well

# In[116]:


# Create a dataframe for Ridge Coefficients
mod_ridge = list(zip(cols, model_parameters))


# In[117]:


paraRFE = pd.DataFrame(mod_ridge)
paraRFE.columns = ['Variable', 'Coeff']
res=paraRFE.sort_values(by=['Coeff'], ascending = False)
res.head(20)


# In[118]:


# Sorting the coefficients in ascending order
paraRFE = paraRFE.sort_values((['Coeff']), axis = 0, ascending = False)
paraRFE


# In[119]:


## since there were few coefficients at 0, we removed them from features
predRFE = pd.DataFrame(paraRFE[(paraRFE['Coeff'] != 0)])
predRFE


# In[120]:


predRFE.shape


# #
# Observation:
#     
# Though the model performance by Ridge Regression was better in terms of R2 values of Train and Test,
# 
# it is better to use Lasso, since it brings and assigns a zero value to insignificant features, enabling us to choose
# the predictive variables.
# 
# It is always advisable to use simple yet robust model.
# 
# Equation can be formulated using the features and coefficients obtained by Lasso

# In[122]:


### Assign the Features as x1, x2.....

pred.set_index(pd.Index(['C','x1', 'x2', 'x3', 'x4', 'x5' , 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']), inplace = True) 
pred


# #
# These are the final features that should be selected for predicting the price of house
# 
# Hence the equation:
# Log(Y) = C + 0.125(x1) + 0.112(x2) + 0.050(x3) + 0.042(x4) + 0.035(x5) + 0.034(x6) + 0.024(x7) + 0.015(x8) + 0.014(x9) + 0.010(x10)
#  +0.010(x11) + 0.005(x12) - 0.007(x13) - 0.007(x14) - 0.008(x15) - 0.095(x16) + Error term(RSS + alpha * (sum of absolute value of coefficients)

# #
# NFERENCE
# 
# Suggestions for Surprise Housing is to keep a check on these predictors affecting the price of the house.
# 
# The higher values of positive coeeficients suggest a high sale value.
# 
# The higher values of negative coeeficients suggest a decrease in sale value.
# 

# In[ ]:




