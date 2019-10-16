from scipy import stats
import numpy as np
from load import load_data
from sklearn.linear_model import LogisticRegression
from model import re_accuracy, dp

def co_with_y():
   data = load_data(False)
   x, y = data['x'], data['y']
   corr = []
   for i in x.T:
      corr.append(abs(stats.pearsonr(i, y.reshape(-1))[0]))
   sorting = np.argsort(corr)[-10:]
   for j in sorting:
       print(get_names()[j])


def co_with_a():
    data = load_data(False)
    x, a = data['x'], data['a']
    corr = []
    for i in x.T:
        corr.append(abs(stats.pearsonr(i, a.reshape(-1))[0]))
    sorting = np.argsort(corr)[-10:]
    for j in sorting:
        print(get_names()[j])
    return sorting

def get_names():
    file = open('adult/adult_headers.txt')
    return file.read().split()



def log_train():
    test_data = load_data(True)
    train_data = load_data(False)
    x, a = train_data['x'], train_data['a']
    a1 = test_data['a']
    clf = LogisticRegression(C= 1000).fit(train_data['x'], train_data['y'].reshape(-1))
    print('logr egression acc',clf.score(test_data['x'], test_data['y'].reshape(-1)))
    print('reweighed acc for log regression',re_accuracy(test_data['a'].reshape(-1), clf.predict(test_data['x']), test_data['y'].reshape(-1)))
    print('delta dp', dp(test_data['a'].reshape(-1), clf.predict(test_data['x'])))
    print((clf.predict(test_data['x'][a1 == 0]).mean()))
    print((clf.predict(test_data['x'][a1 == 1]).mean()))


    corr = []
    for i in x.T:
        corr.append(abs(stats.pearsonr(i, a.reshape(-1))[0]))
    sorted_corr = np.argsort(corr)[-10:]

    indx = [j for j in range(len(x.T)) if j not in sorted_corr]
    train_data['x'] = train_data['x'][:,indx]
    test_data['x'] = test_data['x'][:,indx]
    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['y'].reshape(-1))
    print('accuracy after remove the most correlated with A', clf.score(test_data['x'], test_data['y'].reshape(-1)))
    print('reweighed acc for log regression remove corr with A',re_accuracy(test_data['a'].reshape(-1), clf.predict(test_data['x']), test_data['y'].reshape(-1)))
    print('delta dp remove corr with A',dp(test_data['a'].reshape(-1), clf.predict(test_data['x'])))
    print((clf.predict(test_data['x'][a1 == 0]).mean()))
    print((clf.predict(test_data['x'][a1 == 1]).mean()))


def predict_corr():
    test_data = load_data(True)
    train_data = load_data(False)
    x, a = test_data['x'], test_data['a']
    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['y'].reshape(-1))
    y_pred = clf.predict(test_data['x'])

    corr = []
    for i in x.T:
        replace_nan = abs(stats.pearsonr(i, y_pred)[0])
        if np.isnan(replace_nan):
            corr.append(0)
        else :
            corr.append(replace_nan)

    sorted_corr = np.argsort(corr)[-3:]
    print(sorted_corr)
    for j in sorted_corr:
        print(get_names()[j])


def fe_filtered_corr():
    test_data = load_data(True)
    train_data = load_data(False)
    a = test_data['a']
    x = test_data['x'][a == 0]
    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['y'].reshape(-1))
    y_pred = clf.predict(x)

    corr = []
    for i in x.T:
        replace_nan = abs(stats.pearsonr(i, y_pred)[0])
        if np.isnan(replace_nan):
            corr.append(0)
        else :
            corr.append(replace_nan)

    sorted_corr = np.argsort(corr)[-3:]
    print(sorted_corr)
    for j in sorted_corr:
        print(get_names()[j])


def ma_filtered_corr():
    test_data = load_data(True)
    train_data = load_data(False)
    a = test_data['a']
    x = test_data['x'][a == 1]
    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['y'].reshape(-1))
    y_pred = clf.predict(x)

    corr = []
    for i in x.T:
        replace_nan = abs(stats.pearsonr(i, y_pred)[0])
        if np.isnan(replace_nan):
            corr.append(0)
        else :
            corr.append(replace_nan)

    sorted_corr = np.argsort(corr)[-3:]
    print(sorted_corr)
    for j in sorted_corr:
        print(get_names()[j])


def removed_columns():
    test_data = load_data(True)
    train_data = load_data(False)
    x, a = test_data['x'], test_data['a']

    columns_rm = [get_names().index('sex_Female'), get_names().index('sex_Male')]
    indx = [j for j in range(len(x.T)) if j not in columns_rm]
    train_data['x'] = train_data['x'][:, indx]
    test_data['x'] = test_data['x'][:, indx]
    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['a'].reshape(-1))
    print(clf.score(test_data['x'], test_data['a'].reshape(-1)))
    print(re_accuracy(test_data['a'].reshape(-1), clf.predict(test_data['x']), test_data['a'].reshape(-1)))

def removed_columns_corr():
    test_data = load_data(True)
    train_data = load_data(False)
    x, a = test_data['x'], test_data['a']

    columns_rm = [get_names().index('sex_Female'), get_names().index('sex_Male')] + list(co_with_a())
    indx = [j for j in range(len(x.T)) if j not in columns_rm]
    train_data['x'] = train_data['x'][:, indx]
    test_data['x'] = test_data['x'][:, indx]

    clf = LogisticRegression(C=1000).fit(train_data['x'], train_data['a'].reshape(-1))
    print(clf.score(test_data['x'], test_data['a'].reshape(-1)))
    print(re_accuracy(test_data['a'].reshape(-1), clf.predict(test_data['x']), test_data['a'].reshape(-1)))




def q1():
    test_data = load_data(True)
    #co_with_y()
    co_with_a()

def q2():
    log_train()

def q3():
    predict_corr()
    fe_filtered_corr()
    ma_filtered_corr()
def q4():
    removed_columns()
    removed_columns_corr()
if __name__ == "__main__":
   #q1()
   q2()
   #q3()
   #q4()

