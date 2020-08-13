import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import exp, log
from time import perf_counter
import csv
from os import mkdir

def load_params():
    A = pd.read_csv('models/ar/A.csv', header=None).to_numpy()
    B = pd.read_csv('models/ar/B.csv', header=None).to_numpy()
    output_weights = pd.read_csv('models/ar/output_weights.csv', header=None).to_numpy()
    return A, B, output_weights

def init_params():
    # TODO consider adding input weights
    #input_weights = np.random.normal(size=[input_size, hidden_size])
    # should these be normal or rand?
    A = np.random.normal(size=[hidden_size, input_size])
    B = np.random.normal(size=[hidden_size])
    return A, B

def hidden_nodes(X):  # be careful with variable name
    output = [[0]*hidden_size for i in range(len(X))]
    for i in range(len(X)):
        for j in range(hidden_size):
            temp = np.linalg.norm(A[j] - X.iloc[i])
            temp = temp**2
            temp = temp/(B[j]**2)
            temp = exp(-temp)
            output[i][j] = temp
    return np.array(output)

def train():
    start = perf_counter()
    #training model
    G = hidden_nodes(X_train)
    K = np.dot(G.T, G)
    K = np.linalg.pinv(K)
    t = np.dot(K, G.T)
    output_weights = np.dot(t, y_train)
    end = perf_counter()
    print('Time taken to train in seconds: ', end-start)
    return output_weights
X = pd.read_csv("data/X_AR.csv")
y = pd.read_csv("data/y_AR.csv")

y = y[2:]
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=False)

print(y_test.shape)
input_size = X_train.shape[1]
hidden_size = 20

should_load_params = False

if should_load_params == True:
    A, B, output_weights = load_params()
else:
    print('training the network...')
    A, B = init_params()
    output_weights = train()


def save_params(version):
    mkdir('./models/'+version)
    name = 'models/' + version + '/output_weights.csv'
    file = open(name, 'w+', newline='')
    writer = csv.writer(file)
    with file:
        writer.writerows(output_weights.tolist())

    name = 'models/' + version + '/A.csv'
    file = open(name, 'w+', newline='')
    writer = csv.writer(file)
    with file:
        writer.writerows(A.tolist())

    name = 'models/' + version + '/B.csv'
    file = open(name, 'w+', newline='')
    writer = csv.writer(file)
    with file:
        writer.writerows(B.reshape(-1, 1))


def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out


def remove_outliers(a):
    mean = np.mean(a)
    sd = np.std(a)
    dist_from_mean = abs(a - mean)
    max_deviations = 2
    b = np.copy(a)
    for i in range(len(a)):
        if dist_from_mean[i] > max_deviations * sd:
            if i == 0:
                b[i] = b[i+1]
            elif i == len(a)-1:
                b[i] = b[i-1]
            else:
                b[i] = (b[i-1]+b[i+1])/2
    return b


prediction = predict(X_test)

file = open('data/predictions.csv', 'w+', newline='')
writer = csv.writer(file)
with file:
    writer.writerows(prediction)

prediction_no_outliers = remove_outliers(prediction)
total = X_test.shape[0]
loss = 0
loss2 = 0
for i in range(total):
    predicted = prediction[i]
    actual = y_test.iloc[i].values[0]
    predicted2 = prediction_no_outliers[i]
    diff2 = abs(actual-predicted2)
    loss2 += diff2
    diff = abs(actual-predicted)
    loss += diff
loss = -log(loss/total)
loss2 = -log(loss2/total)
print('Loss for', hidden_size, 'hidden nodes: ', loss)
print('Loss for', hidden_size, 'hidden nodes with outliers removed: ', loss2)

if should_load_params != True:
    choice = ''
    while choice != 'y' and choice != 'n':
        choice = input('Do you wish to save the parameters for this trained model? (y/n) ')

    if choice == 'y':
        choice = input('Enter version no. to save as: ')
        save_params(choice)
        print('parameters saved.')
    elif choice == 'n':
        print('paramaters discarded.')
