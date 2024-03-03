from sklearn import neural_network
import pandas as pd
import numpy as np


def getAccurate(predictii,etichete_test):
    cnt = 0;
    for i in range(len(my_test_data)):
        if(predictii[i]==etichete_test[i]):
            cnt = cnt+1
    return cnt

# INCARCARE DATE
data_test = pd.read_csv('SPECTF.test')
data_train = pd.read_csv('SPECTF.train')

my_train_data=data_test.to_numpy()
my_test_data=data_test.to_numpy()
etichete_train=my_train_data[:,0]
etichete_test=my_test_data[:,0]
my_train_data = np.delete(my_train_data,0,1)
my_test_data = np.delete(my_test_data,0,1)
values_learning_rate=[0.1, 0.01]
hidden_l=[1, 2]
#44
set_of_layers=np.array([[44,44],[44,22],[22,22],[22,11]])
for i in range(len(values_learning_rate)):
    for j in range(len(hidden_l)):
        print('\nRetea neuronala cu ' + str(hidden_l[j]) +' straturi ascunse , '+ str(values_learning_rate[i])+' learning rate\n')
        if hidden_l[j]==1:
            for p in [44,22]:
                clss = neural_network.MLPClassifier(hidden_layer_sizes=p,learning_rate_init=values_learning_rate[i], max_iter=200)
                clss.fit(my_train_data, etichete_train)
                
                #Se calculeaza eroarea
                predictii = clss.predict(my_test_data)
                eroare = getAccurate(predictii,etichete_train)
                result = (eroare/len(etichete_test))*100
                print('\nAcuratetea pentru '+str(p)+' neuroni pe strat este de '+ str("{:.2f}".format(result))+'%') 
        if hidden_l[j]==2:
            for p in [10,5]:
                clss = neural_network.MLPClassifier(hidden_layer_sizes=set_of_layers[p],learning_rate_init=values_learning_rate[i], max_iter=200)
                clss.fit(my_train_data, etichete_train)
                
                predictii = clss.predict(my_test_data)
                eroare = getAccurate(predictii,etichete_train)
                result = (eroare/len(etichete_test))*100
                print('\nAcuratetea pentru '+str(set_of_layers[p])+' neuroni pe strat este de '+ str("{:.2f}".format(result)) + '%') 
                   
        