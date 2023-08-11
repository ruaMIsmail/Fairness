import torch
from torch import nn
from torch.optim import Adam
from load import load_data
from classifcation import get_names
import numpy as np
from sklearn.svm import SVC
from model import dp, re_accuracy
from mmd import MMD_torch

def pre_processing():
    test_data = load_data(True)
    train_data = load_data(False)
    a = test_data['a']
    a1 = train_data['a']
    x = train_data['x']
    columns_rm = [get_names().index('sex_Female'), get_names().index('sex_Male')]
    indx = [j for j in range(len(x.T)) if j not in columns_rm]

    train_data['x'] = train_data['x'][:, indx]
    test_data['x'] = test_data['x'][:, indx]

    xtrain_fe = train_data['x'][a1 == 0]
    xtrain_ma = train_data['x'][a1 == 1]

    xtest_fe = test_data['x'][a == 0]
    xtest_ma = test_data['x'][a == 1]

    ytrain_fe = train_data['y'][a1 == 0]
    ytrain_ma = train_data['y'][a1 == 1]

    ytest_fe = test_data['y'][a == 0]
    ytest_ma = test_data['y'][a == 1]

    return ytrain_fe, ytrain_ma, ytest_fe, ytest_ma,xtrain_fe, xtrain_ma, xtest_fe, xtest_ma


def normalized():
    _, _, _, _, xtrain_fe, xtrain_ma, xtest_fe, xtest_ma = pre_processing()
    var1 = xtrain_fe.std(axis = 0)
    var1[np.abs(var1)< 1e-3] = 1

    var2 = xtrain_ma.std(axis = 0)
    var2[np.abs(var2)< 1e-3] = 1

    normalized_fe = (xtrain_fe - xtrain_fe.mean(axis = 0))/var1
    normalized_ma = (xtrain_ma - xtrain_ma.mean(axis = 0))/var2

    normalized_fe_test = (xtest_fe - xtrain_fe.mean(axis = 0))/var1
    normalized_ma_test = (xtest_ma - xtrain_ma.mean(axis=0))/var2

    return normalized_fe, normalized_ma, normalized_fe_test, normalized_ma_test


def kernal_SVM():
    xtrain_fe, xtrain_ma, xtest_fe, xtest_ma = normalized()
    xtrain = np.concatenate([xtrain_fe, xtrain_ma])
    xtest = np.concatenate([xtest_fe, xtest_ma])

    ytrain_fe, ytrain_ma, ytest_fe, ytest_ma, _, _, _, _ = pre_processing()
    ytrain = np.concatenate([ytrain_fe, ytrain_ma])
    ytest = np.concatenate([ytest_fe, ytest_ma])

    clf = SVC().fit(xtrain, ytrain)
    print(clf.score(xtest, ytest.reshape(-1)))
    print(re_accuracy(ytest.reshape(-1), clf.predict(xtest), ytest.reshape(-1)))

def get_data():
    test_data = load_data(True)
    train_data = load_data(False)
    x = train_data['x']
    columns_rm = [get_names().index('sex_Female'), get_names().index('sex_Male')]
    indx = [j for j in range(len(x.T)) if j not in columns_rm]

    train_data['x'] = train_data['x'][:, indx]
    test_data['x'] = test_data['x'][:, indx]
    return train_data['x'], test_data['x'], train_data['y'], test_data['y'], train_data['a'], test_data['a']


def get_data_normalized():
    xtrain_fe, xtrain_ma, xtest_fe, xtest_ma = normalized()
    xtrain = np.concatenate([xtrain_fe, xtrain_ma])
    xtest = np.concatenate([xtest_fe, xtest_ma])

    ytrain_fe, ytrain_ma, ytest_fe, ytest_ma, _, _, _, _ = pre_processing()
    ytrain = np.concatenate([ytrain_fe, ytrain_ma])
    ytest = np.concatenate([ytest_fe, ytest_ma])

    atrain = np.zeros(xtrain_fe.shape[0] + xtrain_ma.shape[0])
    atrain[xtrain_fe.shape[0]:] = 1

    atest = np.zeros(xtest_fe.shape[0] + xtest_ma.shape[0])
    atest[xtest_fe.shape[0]:] = 1

    return xtrain, xtest, ytrain, ytest, atrain, atest


def NN(x, y):
    model = torch.nn.Sequential(nn.Linear(x.shape[1], 100 ),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100,100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100,1)

    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay= 1e-5)

    for e in range(10):
        indx = torch.randperm(x.shape[0])
        x = x[indx]
        y = y[indx]
        batch_size = 512
        for b in range(0, x.shape[0], batch_size):
            x_batch = x[b:b+batch_size]
            y_batch = y[b:b+batch_size]

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def NN_mmd(x, y, a, alpha):
    model1 = torch.nn.Sequential(nn.Linear(x.shape[1], 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                nn.ReLU()

                                 )
    model2 = torch.nn.Sequential(nn.Linear(100,1))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(list(model1.parameters())+list(model2.parameters()), weight_decay=1e-5)

    for e in range(10):
        indx = torch.randperm(x.shape[0])
        x = x[indx]
        y = y[indx]
        a = a[indx]
        batch_size = 512
        for b in range(0, x.shape[0], batch_size):
            x_batch = x[b:b + batch_size]
            y_batch = y[b:b + batch_size]
            a_batch = a[b:b + batch_size]
            features = model1(x_batch)
            y_pred = model2(features)
            loss = loss_fn(y_pred, y_batch) + alpha * MMD_torch(features, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model1, nn.Sequential(model1, model2)


def accuracy_(model, x, y):
    x = torch.tensor(x).float()
    y = torch.tensor(y).byte()
    pred = model(x) > 0
    n = x.shape[0]
    acc = (pred.view(-1) == y.view(-1)).float().sum() * 100 / n
    return acc

def q1():
    xtrain, xtest, ytrain, ytest, atrain, atest = get_data_normalized()
    xtrain = torch.tensor(xtrain).float()
    ytrain = torch.tensor(ytrain[:,None]).float()
    atrain = torch.tensor(atrain[:,None]).float()

    xtest = torch.tensor(xtest).float()
    ytest = torch.tensor(ytest[:,None]).float()
    atest = torch.tensor(atest[:,None]).float()



    model1 = NN(xtrain, ytrain)
    acc = accuracy_(model1, xtest, ytest)
    print('model y normalized acc:', acc)
    print('model y dp :', dp(atrain.int().numpy().ravel(), (model1(xtrain)>0).numpy().ravel()))

    model2 = NN(xtrain, atrain)
    acc = accuracy_(model2, xtest, atest)
    print('model a normalized reweighed acc:',re_accuracy(atest.numpy().reshape(-1), (model2(xtest)>0).numpy().reshape(-1), atest.numpy().reshape(-1)))
    print('model a acc:', acc)

    x_train, x_test, y_train, y_test, a_train, a_test = get_data()
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train[:, None]).float()
    a_train = torch.tensor(a_train[:, None]).float()

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test[:, None]).float()
    a_test = torch.tensor(a_test[:, None]).float()

    model3 = NN(x_train, y_train)
    acc = accuracy_(model3, x_test, y_test)
    print('model y un-normalized acc:',acc)
    print('model y un-normalized dp:', dp(a_train.int().numpy().ravel(), (model3(x_train)>0).numpy().ravel()) )

    model4 = NN(x_train, a_train)
    acc = accuracy_(model4, x_test, a_test)
    print('model a un-normalized acc', acc)
    print('model a un-normalized reweighed acc:', re_accuracy(a_test.numpy().reshape(-1), (model4(x_test)>0).numpy().reshape(-1), a_test.numpy().reshape(-1)))


def q2():
    data_train = load_data(False)
    xtrain = data_train['x']
    ytrain = data_train['y']
    atrain = data_train['a']

    data_test = load_data(True)
    xtest = data_test['x']
    ytest = data_test['y']
    atest = data_test['a']

    xtrain = torch.tensor(xtrain).float()
    ytrain = torch.tensor(ytrain[:, None]).float()
    atrain = torch.tensor(atrain[:, None]).float()

    xtest = torch.tensor(xtest).float()
    ytest = torch.tensor(ytest[:, None]).float()
    atest = torch.tensor(atest[:, None]).float()

    features_extractor, classifers = NN_mmd(xtrain, ytrain, atrain, 0.1)
    acc = accuracy_(classifers, xtest, ytest)
    print('acc for MMD NN: ',acc)
    print('dp for features:', dp(atrain.int().numpy().ravel(), (classifers(xtrain)>0).numpy().ravel()))

    xtrain_feature = features_extractor(xtrain).detach()
    xtest_feature = features_extractor(xtest).detach()


    model2 = NN(xtrain_feature, atrain)
    acc = accuracy_(model2, xtest_feature, atest)
    print('model a normalized reweighed acc:',
          re_accuracy(atest.numpy().reshape(-1), (model2(xtest_feature) > 0).numpy().reshape(-1), atest.numpy().reshape(-1)))
    print('model a acc:', acc)

def q3():
    data_train = load_data(False)
    xtrain = data_train['x']
    ytrain = data_train['y']
    atrain = data_train['a']

    data_test = load_data(True)
    xtest = data_test['x']
    ytest = data_test['y']
    atest = data_test['a']

    xtrain = torch.tensor(xtrain).float()
    ytrain = torch.tensor(ytrain[:, None]).float()
    atrain = torch.tensor(atrain[:, None]).float()

    xtest = torch.tensor(xtest).float()
    ytest = torch.tensor(ytest[:, None]).float()
    atest = torch.tensor(atest[:, None]).float()
    accuracy = []
    d_p = []
    alphas = [.01, .1, 1, 10, 100]
    for alpha in alphas:
        print('Alpha :', alpha)
        features_extractor, classifers = NN_mmd(xtrain, ytrain, atrain, alpha)
        acc = accuracy_(classifers, xtest, ytest)
        accuracy.append(acc)
        print('acc for MMD NN: ', acc)
        delta_dp = dp(atrain.int().numpy().ravel(), (classifers(xtrain) > 0).numpy().ravel())
        d_p.append(delta_dp)
        print('dp for features:', delta_dp)
    print('accuracy ', accuracy)
    print('alpha', alphas)
    print('d_p', d_p)

# accuracy  [tensor(82.5625), tensor(83.0784), tensor(83.9998), tensor(79.4669), tensor(79.9275)]
# alpha [0.01, 0.1, 1, 10, 100]
# d_p [0.15164437729870492, 0.176766489522863, 0.1579677153085029, 0.06059384979358125, 0.038998485258356745]




if __name__ == "__main__":
    q1()

    pass
