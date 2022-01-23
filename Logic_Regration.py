# Initializing the dataframe
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('D:/Machine Learing/iris_trainer_trster/iris.csv')



# See head of the dataset
#df.plot(kind = 'scatter',x = 'sepal.length', y= 'sepal.width')
#
# Horizental bar 

# 
# df["Duration"].plot(kind = 'hist')
#print(dataset)

# make x independent variable
x=dataset.iloc[:,0:4].values
#print( dataset.iloc[:,0:4].values)
#print(x)

y = dataset.iloc[:,4].values
#print(y)
# change text in numaric form
# 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

# logistic regration 



from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
y_pred = logmodel.predict(x_test)
#print(y_pred,y_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
# linear regration 
# multipal regration
# polynomyal regration 

# naive bayes



# k near classification 
'''
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric=  'minkowski', p=2)
classifier_knn.fit(x_train,y_train)
y_pred = classifier_knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))


# support vector machine 

# sigmoid


from sklearn.svm import SVC
classifier_svm_sigmoid = SVC(kernel='sigmoid')
classifier_svm_sigmoid.fit(x_train,y_train)
y_pred = classifier_svm_sigmoid.predict(x_test)
print(confusion_matrix(y_test,y_pred))


# linear 

from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel='linear')
classifier_svm_linear.fit(x_train,y_train)
y_pred = classifier_svm_linear.predict(x_test)
print(confusion_matrix(y_test,y_pred))


# rbf


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel='rbf')
classifier_svm_rbf.fit(x_train,y_train)
y_pred = classifier_svm_rbf.predict(x_test)
print(confusion_matrix(y_test,y_pred))


#  poly


from sklearn.svm import SVC
classifier_svm_poly = SVC(kernel='poly')
classifier_svm_poly.fit(x_train,y_train)
y_pred = classifier_svm_poly.predict(x_test)
print(confusion_matrix(y_test,y_pred))


# disigion tree 

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion= 'entropy')
classifier_dt.fit(x_train,y_train)
y_pred = classifier_dt.predict(x_test)
print(confusion_matrix(y_test,y_pred))



# random forest 


from sklearn.ensemble import RandomForestClassifier 
classifier_rf=RandomForestClassifier(n_estimators=3 , criterion= 'entropy')
classifier_rf.fit(x_train,y_train)
y_pred = classifier_rf.predict(x_test)
print(confusion_matrix(y_test,y_pred))'''

# nural natwork 

# clustering 
# k mean 
#hierarchical
# mean shift
# density Based
# dimensionality Reduction
# fecature eleminatlion
# feature extraction 
# PCA 