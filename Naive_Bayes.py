import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("promoters.data", header = None, names = ['Class','Instance','Sequence'])
seq_list = df.iloc[:,2].str.strip()

seq_array = []
for lst in seq_list:
    seq_array.append(list(lst))

df_t = pd.DataFrame(seq_array, columns = ['S'+str(i) for i in xrange(1,58)])
df_t = df_t.replace('a',1)
df_t = df_t.replace('g',2)
df_t = df_t.replace('c',3)
df_t = df_t.replace('t',4)

df = pd.concat([df['Instance'],df['Class'],df_t],axis=1)
df.set_index('Instance', inplace=True)

m,n = df.shape

t70 = int(m * 0.7)
train = np.random.choice(m,t70,replace=False)

train_data = df.iloc[train,:]
test_data = df.drop(df.index[train])

print "Train Data Shape : ",train_data.shape
print "Test Data Shape : ",test_data.shape

gnb = GaussianNB()
gnb = gnb.fit(train_data.iloc[:,1:], train_data.iloc[:,0])

print "Training Phase :: ",

y_pred = gnb.predict(train_data.iloc[:,1:])
conf_mat = confusion_matrix(train_data.iloc[:,0],y_pred)

print "Confusion Matrix : "
print conf_mat
print "Accuracy : ",float(conf_mat[0,0]+conf_mat[1,1]) / np.sum(conf_mat)

print "Testing Phase :: ",

y_pred = gnb.predict(test_data.iloc[:,1:])
conf_mat = confusion_matrix(test_data.iloc[:,0],y_pred)

print "Confusion Matrix : "
print conf_mat
print "Accuracy : ",float(conf_mat[0,0]+conf_mat[1,1]) / np.sum(conf_mat)
