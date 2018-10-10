#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import os


# In[2]:


from scipy import stats
import researchpy as rp
from sklearn.preprocessing import PowerTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost as xgb


# In[3]:


train=pd.read_csv("C://Users//User//Documents//Python_Projects//kaggle_fct//datasets//train.csv")
test=pd.read_csv("C://Users//User//Documents//Python_Projects//kaggle_fct//datasets//test.csv")


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


test.head()


# In[7]:


#숫자형 변수만 따로 추출한다.
train_num=train.iloc[:,1:11]
test_num=test.iloc[:,1:11]


# In[8]:


#정규분포 맞춰주기+표준화
pt=PowerTransformer()
pt.fit(train_num)
train_trans=pt.transform(train_num)
pt.fit(test_num)
test_trans=pt.transform(test_num)


# In[9]:


train_num_df=pd.DataFrame(train_trans, columns=train_num.columns)
train_num_df.head()


# In[10]:


test_num_df=pd.DataFrame(test_trans, columns=test_num.columns)
test_num_df.head()


# In[11]:


#분포를 확인한다.
g = sns.FacetGrid(train_num_df.melt(), col="variable", col_wrap=3, aspect=2)
g.map(sns.distplot,"value")

plt.show()


# In[12]:


#훈련데이터 정렬?
train_x=train_num_df
train_y=train["Cover_Type"]

#테스트데이터 정렬?
test_x=test_num_df


# In[13]:


#모델을 훈련시킨다.
ovr_clf=OneVsRestClassifier(RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523))
ovr_scores = cross_val_score(ovr_clf, train_x, train_y, cv=10)
print('Scores =', ovr_scores)


# In[14]:


print('CV accuracy: %.3f +/- %.3f' % (np.mean(ovr_scores), np.std(ovr_scores)))


# In[15]:


#모델을 훈련시킨다.
rf_clf=RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523)
rf_scores = cross_val_score(rf_clf, train_x, train_y, cv=10)
print('Scores =', rf_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(rf_scores), np.std(rf_scores)))


# In[16]:


rf_clf.fit(train_x,train_y)


# In[17]:


rf_clf.feature_importances_


# In[18]:


feat_importances = pd.Series(rf_clf.feature_importances_, index=train_x.columns)

plt.rcParams['figure.figsize'] = [15, 8]
feat_importances.sort_values().plot(kind='barh')
plt.savefig('rf_num_varImp.png')


# In[19]:


#명목형 변수를 살펴본다.
train_cat=train.iloc[:,11:56]
train_cat_df=pd.DataFrame(train_cat, columns=train_cat.columns)


# In[20]:


train_cat_df.head()


# In[21]:


train_table=train_cat_df.groupby(["Cover_Type"]).sum()
train_table


# In[22]:


m = pd.melt(train_cat_df, id_vars=['Cover_Type'], var_name='Type')
m.head()


# In[23]:


g_sum=m.groupby(['Cover_Type','Type'])['value'].sum()

plt.rcParams['figure.figsize'] = [15, 8]
g_sum.nlargest(25).plot(kind="barh")

plt.savefig('cat_eda.png')


# In[24]:


#테스트 변수의 명목형 변수 묶음을 만들어준다.
test_cat=test.iloc[:,11:55]
test_cat_df=pd.DataFrame(test_cat, columns=test_cat.columns)


# In[25]:


ctype_type = pd.crosstab(index=train_cat_df["Cover_Type"], 
                            columns=train_cat_df["Wilderness_Area1"])
ctype_type


# In[26]:


stats.chi2_contingency(ctype_type)


# In[27]:


table, results = rp.crosstab(train_cat_df["Cover_Type"], train_cat_df["Wilderness_Area1"], prop= 'col', test= 'chi-square')
table


# In[28]:


print(results)


# In[29]:


for col in train_cat_df.columns[:-1]:
    table, results = rp.crosstab(train_cat_df["Cover_Type"], train_cat_df[col], prop= 'col', test= 'chi-square')
    if(results.results[1]<0.05):
        print(col,"의 p-value는 ",results.results[1],"입니다.","\n")


# In[30]:


train_cat_x=train_cat_df.iloc[:,:-1]
train_cat_y=train_cat_df.iloc[:,-1]


# In[31]:


clf_cat_rf=RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523)
cat_rf_scores = cross_val_score(clf_cat_rf,train_cat_x, train_cat_y, cv=10)


# In[32]:


print('Scores =', cat_rf_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(cat_rf_scores), np.std(cat_rf_scores)))


# In[33]:


clf_cat_rf.fit(train_cat_x, train_cat_y)


# In[34]:


clf_cat_rf.feature_importances_


# In[35]:


feat_importances = pd.Series(clf_cat_rf.feature_importances_, index=train_cat_x.columns)

plt.rcParams['figure.figsize'] = [15, 8]
feat_importances.sort_values().nlargest(20).plot(kind='barh')
plt.savefig('rf_cat_varImp.png')


# In[36]:


#사전분석을 통한 결과를 토대로 변수를 선택한다.
x_variable_num=["Elevation","Horizontal_Distance_To_Roadways","Hillshade_9am","Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Fire_Points"]
x_variable_cat=["Wilderness_Area4","Wilderness_Area1","Wilderness_Area3","Soil_Type10","Soil_Type38","Soil_Type39"]
train_num_df[x_variable_num].reset_index(drop=True, inplace=True)
train_cat_df[x_variable_cat].reset_index(drop=True, inplace=True)

train_x_xg=pd.concat([train_num_df[x_variable_num],train_cat_df[x_variable_cat]],axis=1)

train_num_df.reset_index(drop=True, inplace=True)
train_cat_df.iloc[:,:-1].reset_index(drop=True, inplace=True)

train_x_trans_cat=pd.concat([train_num_df,train_cat_df.iloc[:,:-1]],axis=1)


# In[37]:


train_x_xg.head()


# In[38]:


train_x_trans_cat.head()


# In[39]:


train_y_xg=train["Cover_Type"]


# In[41]:


#변수선택을 직접한 모델
xgb_model = xgb.XGBClassifier()
kfold = KFold(n_splits=10, random_state=523)
manual_results = cross_val_score(xgb_model, train_x_xg, train_y_xg, cv=kfold)
print('CV accuracy: %.3f +/- %.3f' % (manual_results.mean()*100, manual_results.std()*100))


# In[42]:


#변수선택을 하지 않은 모델
results = cross_val_score(xgb_model, train_x_trans_cat, train_y_xg, cv=kfold)
print('CV accuracy: %.3f +/- %.3f' % (results.mean()*100, results.std()*100))


# In[44]:


#랜텀 포레스트와도 비교/변수선택
clf_rf=RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523)
rf_manual_scores = cross_val_score(clf_rf,train_x_xg, train_y_xg, cv=10)
print('Scores =', rf_manual_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(rf_manual_scores), np.std(rf_manual_scores)))


# In[45]:


#랜텀 포레스트와도 비교/모든 변수
rf_scores = cross_val_score(clf_rf,train_x_trans_cat, train_y_xg, cv=10)
print('Scores =', rf_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(rf_scores), np.std(rf_scores)))


# In[47]:


clf_rf.fit(train_x_trans_cat, train_y_xg)


# In[48]:


test_num_df.reset_index(drop=True, inplace=True)
test.iloc[:,11:].reset_index(drop=True, inplace=True)

test_trans_cat=pd.concat([test_num_df,test.iloc[:,11:]],axis=1)


# In[52]:


pred_y_rf=clf_rf.predict(test_trans_cat)


# In[54]:


sub=pd.DataFrame(test.iloc[:,0])
sub["Cover_Type"]=pred_y_rf


# In[55]:


sub.to_csv("submission.csv",index=False)

