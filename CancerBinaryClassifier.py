import sklearn 

# input data
from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer() 
label_names = data['target_names'] 
labels = data['target'] 
feature_names = data['feature_names'] 
features = data['data'] 

# splitting data into train and test sets
from sklearn.model_selection import train_test_split 
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42) 

# import binary classifier and fit
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
model = gnb.fit(train, train_labels) 

# make predictions 
predictions = gnb.predict(test) 
# print(predictions) 

# measure performances
from sklearn.metrics import accuracy_score 
print(accuracy_score(test_labels, predictions)) 
