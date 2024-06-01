def Split(X, y, split):
    X_train = X[:(int(len(X)*split)-1)]
    y_train = y[:(int(len(y)*split)-1)]
    X_predict = X[(int(len(X)*split)-1):]
    y_predict = y[(int(len(y)*split)-1):]
    return X_train, y_train, X_predict, y_predict