

# Initializing the dataframe
import pandas as pd
# plot graph..
import matplotlib.pyplot as plt
# transform text into numaric form 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
#split the data 
from sklearn.model_selection import train_test_split
# shoe the result 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 

# lode the data 
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


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel='linear')
classifier_svm_linear.fit(x_train,y_train)
y_pred = classifier_svm_linear.predict(x_test)
print(confusion_matrix(y_test,y_pred))
