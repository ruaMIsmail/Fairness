from sklearn.linear_model import LogisticRegression
import numpy as np

def log_train(x, y):

    clf = LogisticRegression(C= 1000).fit(x, y)
    return clf


def re_accuracy(a, prediction, label):
    prediction = [prediction[a==0], prediction[a==1]]
    label = [label[a==0], label[a==1]]
    acc = 0
    for pred, label in zip(prediction,label):
        n = pred.size
        acc += (pred == label).astype(np.float64).sum() *100/n
    return acc/2


def dp(a, prediction):
    prediction1 = prediction[a==0].mean()
    prediction2 = prediction[a==1].mean()

    abs_def = abs(prediction1-prediction2)

    return abs_def


if __name__ == "__main__":
    from load import load_data
    train_data = load_data(False)
    trained = log_train(train_data['x'], train_data['y'].reshape(-1))

    test_data = load_data(True)
    print(trained.score(test_data['x'], test_data['y'].reshape(-1)))
    print(re_accuracy(test_data['a'].reshape(-1), trained.predict(test_data['x']), test_data['y'].reshape(-1)))
    print(dp(test_data['a'].reshape(-1), trained.predict(test_data['x'])))
