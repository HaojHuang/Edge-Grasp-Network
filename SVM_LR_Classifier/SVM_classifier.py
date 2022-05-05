# This program is a logistic regression classifier.

import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, balanced_accuracy_score
from datetime import datetime
import pickle

#  ---------------------------------------------------------------------
# Create dummy training dataset
features_trn = 2 * np.concatenate((np.ones((590, 128)),
                                   np.zeros((10, 128)),
                                   np.zeros((590, 128)),
                                   np.ones((10, 128))))  # 600 for class 1, 600 for class 0 and some mislabeled.
features_trn += np.random.normal(0, 0.01, size=features_trn.shape)  # add normal distribution bias.
labels_trn = np.concatenate((np.ones(600, dtype=np.compat.long),
                             np.zeros(600, dtype=np.compat.long)))  # tags for X.

# Create dummy test dataset
features_tst = 2 * np.concatenate((np.ones((600, 128)),
                                   np.zeros((600, 128))))  # 600 for class 1, 600 for class 0
features_tst += np.random.normal(0, 0.01, size=features_tst.shape)  # add normal distribution bias.
labels_tst = np.concatenate((np.ones(600, dtype=np.compat.long),
                             np.zeros(600, dtype=np.compat.long)))  # tags for X.


# This function would train SVM model
# and print the accuracy with respect to the given test data set.
def SVM_classifier(X_trn, Y_trn, X_tst, Y_tst, kernel, model_name, train=True):
    # Get a trained model
    clf = None
    if train is True:
        print(f'{datetime.now()} : Training the model with a {kernel} kernel...')
        # Training...
        clf = svm.SVC(kernel=kernel, class_weight='balanced')
        # clf = svm.LinearSVC(C=1e5, class_weight='balanced', max_iter=10000)
        clf.fit(X_trn, Y_trn)
    else:
        print(f'{datetime.now()} : Loading the existing SVM model...')
        clf = pickle.load(open(model_name, 'rb'))

    print(f'{datetime.now()} : Model prepared! ')

    # the score of this training.
    print(f'{datetime.now()} : Testing model...')
    score_tst = clf.score(X_tst, Y_tst)
    # positive_tst = score_tst>0.5
    # print('positive prediction number', sum(positive_tst))

    # f1 score.
    Y_predict = clf.predict(X_tst)

    positive_bool = Y_predict > 0.5
    print('positive prediction number', sum(positive_bool))
    print(f'total num of test:{len(X_tst)}')
    f1 = f1_score(Y_tst, Y_predict, zero_division=1)
    balanced = balanced_accuracy_score(Y_tst, Y_predict)
    print(f'{datetime.now()} : Testing complete! ')

    print('Balanced accuracy score of the SVM model on the {} test cases: {} '.format(len(Y_tst), balanced))
    print('F1_score of the SVM model on the {} test cases: {} '.format(len(Y_tst), f1))
    print('Accuracy of the SVM model on the {} test cases: {} %\n'.format(len(Y_tst), 100 * score_tst))

    # save the model to disk
    filename = model_name
    pickle.dump(clf, open(filename, 'wb'))

# SVM_classifier(features_trn, labels_trn, features_tst, labels_tst, 'linear')
# SVM_classifier(features_trn, labels_trn, features_tst, labels_tst, 'rbf')
# SVM_classifier(features_trn, labels_trn, features_tst, labels_tst, 'poly')


# load the model from disk
# filename = 'SVM_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(features_tst, labels_tst)
# print(f'the score of the reload model is {result}')