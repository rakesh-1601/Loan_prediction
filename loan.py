import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,SelectPercentile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC,LinearSVC

#import dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train_original=train.copy()
test_original=test.copy()

#Filling empty values
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


#Normalizing LoanAmount data
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])


#creating new feature
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']

train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])


train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)


#Dropping irrelevent features
train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)



train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

y = train.Loan_Status
train=train.drop('Loan_Status',axis=1)
train=pd.get_dummies(train)
test=pd.get_dummies(test)


train['Individual']=train['Dependents_1']+train['Dependents_2']+train['Dependents_3+']
test['Individual']=test['Dependents_1']+test['Dependents_2']+test['Dependents_3+']


train=train.drop(['Gender_Male','Gender_Female','Property_Area_Urban','Total_Income_log','Property_Area_Semiurban','Property_Area_Rural','Dependents_1', 'Dependents_2', 'Dependents_3+','Dependents_0','Individual','Married_Yes','Married_No','Self_Employed_Yes','Self_Employed_No','Education_Graduate','Education_Not Graduate'], axis=1)
test=test.drop(['Gender_Male','Gender_Female','Property_Area_Urban','Total_Income_log','Property_Area_Semiurban','Property_Area_Rural','Dependents_1', 'Dependents_2', 'Dependents_3+','Dependents_0','Individual','Married_Yes','Married_No','Self_Employed_Yes','Self_Employed_No','Education_Graduate','Education_Not Graduate'], axis=1)

X = train
model = XGBClassifier()# Random forest can also be used
#fitting model
model.fit(X,y)
list=[]
i = 1

#K fold cross validation..!!!
kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]
    model = XGBClassifier(random_state=1,n_estimators=76,max_depth=1,)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    list.append(score)
    i += 1
    
#predicting prices    
pred_test = model.predict(test)
print(np.array(list).mean())



#Just a graph of knowing importance of feature as per selected model
importances=pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))

submission=pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)

plt.show()
