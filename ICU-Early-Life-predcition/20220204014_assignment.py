import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



df=pd.read_csv('X_train_2025.csv')

# Top 5 row
print(df.head())
# This is for info of data
print(df.info())
# This is for shape of my data
print(df.shape)
# This describe my data
print(df.describe)

dfy=df['In-hospital_death'].values
df=df.drop('In-hospital_death',axis=1)

# Imputer for replacing empty value with Median
imputer=SimpleImputer(strategy='median')
df=imputer.fit_transform(df)
# Convert to int
df=df.astype(int)

# seperating Test And Train Data
xtest,xtrain,ytest,ytrain=train_test_split(df,dfy,test_size=0.2,random_state=42)

print("--------------- Decision Tree Classifier -----------------")
decisiontree=DecisionTreeClassifier()
decisiontree=decisiontree.fit(xtrain,ytrain)
yprediction=decisiontree.predict(xtest) 
AccurrayScore=accuracy_score(ytest,yprediction)
PrecisionScore=precision_score(ytest,yprediction)
RecallScore=recall_score(ytest,yprediction)
F1Score=f1_score(ytest,yprediction)
print(f"This is Acuracy with decision tree calssifier  : {AccurrayScore}")
print(f"This is PrecisionScore with decision tree calssifier  : {PrecisionScore}")
print(f"This is RecallScore with decision tree calssifier  : {RecallScore}")
print(f"This is F1Score with decision tree calssifier  : {F1Score}")



print("------------------- Naive Base Classifier -----------------")
naivebaseClassifier=GaussianNB()
naivebaseClassifier=naivebaseClassifier.fit(xtrain,ytrain)
yprediction=naivebaseClassifier.predict(xtest)
AccurrayScore=accuracy_score(ytest,yprediction)
PrecisionScore=precision_score(ytest,yprediction)
RecallScore=recall_score(ytest,yprediction)
F1Score=f1_score(ytest,yprediction)
print(f"This is Acuracy with Navie Base Calssifier  : {AccurrayScore}")
print(f"This is PrecisionScore with Navie Base Calssifier  : {PrecisionScore}")
print(f"This is RecallScore with Navie Base Calssifier  : {RecallScore}")
print(f"This is F1Score with Navie Base Calssifier  : {F1Score}")


print("------------------ KNN Classifier ----------------")
knnclassifire=KNeighborsClassifier()
knnclassifire=knnclassifire.fit(xtrain,ytrain)
yprediction=knnclassifire.predict(xtest)
AccurrayScore=accuracy_score(ytest,yprediction)
PrecisionScore=precision_score(ytest,yprediction)
RecallScore=recall_score(ytest,yprediction)
F1Score=f1_score(ytest,yprediction)
print(f"This is Acuracy with KNN Calssifier  : {AccurrayScore}")
print(f"This is PrecisionScore with KNN Calssifier  : {PrecisionScore}")
print(f"This is RecallScore with KNN Calssifier  : {RecallScore}")
print(f"This is F1Score with KNN Calssifier  : {F1Score}")