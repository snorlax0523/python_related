
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import os


# In[70]:


from scipy import stats
import researchpy as rp
from sklearn.preprocessing import PowerTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost as xgb


# In[3]:


train=pd.read_csv("C://Users/student/Documents/python_projects/simple_projects/jupyter_notebook/kaggle_fct/datasets/train.csv")
test=pd.read_csv("C://Users/student/Documents/python_projects/simple_projects/jupyter_notebook/kaggle_fct/datasets/test.csv")


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


test.head()


# In[8]:


#숫자형 변수만 따로 추출한다.
train_num=train.iloc[:,1:11]
test_num=test.iloc[:,1:11]


# In[9]:


#정규분포 맞춰주기+표준화
pt=PowerTransformer()
pt.fit(train_num)
train_trans=pt.transform(train_num)
pt.fit(test_num)
test_trans=pt.transform(test_num)


# In[10]:


train_num_df=pd.DataFrame(train_trans, columns=train_num.columns)
train_num_df.head()


# In[11]:


test_num_df=pd.DataFrame(test_trans, columns=test_num.columns)
test_num_df.head()


# In[12]:


#분포를 확인한다.
g = sns.FacetGrid(train_num_df.melt(), col="variable", col_wrap=3, aspect=2)
g.map(sns.distplot,"value")

plt.show()


# In[13]:


#훈련데이터 정렬?
train_x=train_num_df
train_y=train["Cover_Type"]

#의존변수를 숫자로 표현할 준비를 한다.
factor = pd.factorize(train['Cover_Type'])
train.Cover_Type=factor[0]
definitions=factor[1]
print(train.Cover_Type.head())
print(definitions)

#테스트데이터 정렬?
test_x=test_num_df


# In[14]:


#모델을 훈련시킨다.
ovr_clf=OneVsRestClassifier(RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523))
ovr_scores = cross_val_score(ovr_clf, train_x, train_y, cv=10)
print('Scores =', ovr_scores)


# In[15]:


print('CV accuracy: %.3f +/- %.3f' % (np.mean(ovr_scores), np.std(ovr_scores)))


# In[16]:


ovr_clf.fit(train_x,train_y)


# In[17]:


#모델을 훈련시킨다.
rf_clf=RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523)
rf_scores = cross_val_score(rf_clf, train_x, train_y, cv=10)
print('Scores =', rf_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(rf_scores), np.std(rf_scores)))


# In[18]:


rf_clf.fit(train_x, train_y)


# In[19]:


#랜덤포레스트에서 중요하게 작용한 수치형 변수를 살펴본다.
rf_clf.feature_importances_


# In[27]:


#그래프로 변수들의 중요도를 살펴본다.
feat_importances = pd.Series(rf_clf.feature_importances_, index=train_x.columns)

plt.rcParams['figure.figsize'] = [15, 8]
feat_importances.sort_values().plot(kind='barh')
plt.savefig('rf_num_varImp.png')


# In[28]:


#명목형 변수를 살펴본다.
train_cat=train.iloc[:,11:56]
train_cat_df=pd.DataFrame(train_cat, columns=train_cat.columns)


# In[29]:


train_cat_df.head()


# In[30]:


train_table=train_cat_df.groupby(["Cover_Type"]).sum()
train_table


# In[31]:


#탐색적 분석을 실시한다.
m = pd.melt(train_cat_df, id_vars=['Cover_Type'], var_name='Type')
m.head()


# In[32]:


g_sum=m.groupby(['Cover_Type','Type'])['value'].sum()

plt.rcParams['figure.figsize'] = [15, 8]
g_sum.nlargest(25).plot(kind="barh")

plt.savefig('cat_eda.png')


# In[33]:


#테스트 변수의 명목형 변수 묶음을 만들어준다.
test_cat=test.iloc[:,11:55]
test_cat_df=pd.DataFrame(test_cat, columns=test_cat.columns)


# In[34]:


ctype_type = pd.crosstab(index=train_cat_df["Cover_Type"], 
                            columns=train_cat_df["Wilderness_Area1"])
ctype_type


# In[35]:


stats.chi2_contingency(ctype_type)


# In[36]:


table, results = rp.crosstab(train_cat_df["Cover_Type"], train_cat_df["Wilderness_Area1"], prop= 'col', test= 'chi-square')
table


# In[37]:


print(results)


# In[38]:


#반복문을 통해 카이제곱 검정을 실시하고 p값을 살펴본다.
for col in train_cat_df.columns[:-1]:
    table, results = rp.crosstab(train_cat_df["Cover_Type"], train_cat_df[col], prop= 'col', test= 'chi-square')
    if(results.results[1]<0.05):
        print(col,"의 p-value는 ",results.results[1],"입니다.","\n")


# In[39]:


train_cat_x=train_cat_df.iloc[:,:-1]
train_cat_y=train_cat_df.iloc[:,-1]


# In[40]:


clf_cat_rf=RandomForestClassifier(n_estimators=211, criterion='entropy', random_state=523)
cat_rf_scores = cross_val_score(clf_cat_rf,train_cat_x, train_cat_y, cv=5)


# In[41]:


print('Scores =', cat_rf_scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(cat_rf_scores), np.std(cat_rf_scores)))


# In[42]:


clf_cat_rf.fit(train_cat_x, train_cat_y)


# In[43]:


clf_cat_rf.feature_importances_


# In[44]:


feat_importances = pd.Series(clf_cat_rf.feature_importances_, index=train_cat_x.columns)

plt.rcParams['figure.figsize'] = [15, 8]
feat_importances.sort_values().nlargest(20).plot(kind='barh')
plt.savefig('rf_cat_varImp.png')


# In[48]:


#사전분석을 통한 결과를 토대로 변수를 선택한다.
x_variable=["Elevation","Horizontal_Distance_To_Roadways","Hillshade_9am","Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Fire_Points","Wilderness_Area4","Wilderness_Area1","Wilderness_Area3","Soil_Type10","Soil_Type38","Soil_Type39"]
train_x_xg=pd.DataFrame(train[x_variable])


# In[49]:


train_x_xg.head()


# In[51]:


train_y_xg=train["Cover_Type"]


# In[77]:


xgb_manual_model = xgb.XGBClassifier()
kfold = KFold(n_splits=10, random_state=523)
manual_results = cross_val_score(xgb_manual_model, train_x_xg, train_y_xg, cv=kfold)
print('CV accuracy: %.3f +/- %.3f' % (manual_results.mean()*100, manual_results.std()*100))


# In[79]:


xgb_model = xgb.XGBClassifier()
kfold = KFold(n_splits=10, random_state=523)
results = cross_val_score(xgb_model, train.iloc[:,1:55], train["Cover_Type"], cv=kfold)
print('CV accuracy: %.3f +/- %.3f' % (results.mean()*100, results.std()*100))


# 수동으로 튜닝한 모델보다 자동 모델이 성능이 더 좋기 때문에 자동 모델을 사용하기로 한다.

# In[88]:


xgb_model.fit(train.iloc[:,1:55], train["Cover_Type"])


# In[178]:


#만들어진 모델을 바탕으로 예측한다.
#모델이 돌아가면서 2개가 날아간다....
pred_y_xg_first=xgb_model.predict(test.iloc[:200000,1:])
pred_y_xg_second=xgb_model.predict(test.iloc[200000:,1:])
pred_y_xg=xgb_model.predict(test.iloc[:,1:])


# In[179]:


#실제 타입은 1~7이기 때문에 1을 더해준 다음 csv로 저장한다.
pred_y_ct_first=pred_y_xg_first+1
pred_y_ct_second=pred_y_xg_second+1
pred_y_ct=pred_y_xg+1


# In[187]:


sub=pd.DataFrame(test.iloc[:,0])
sub["Cover_Type"]=pred_y_ct


# In[188]:


sub.head()


# In[189]:


sub.to_csv("submission.csv")

