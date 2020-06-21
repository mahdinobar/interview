from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
from sklearn.metrics import accuracy_score

def model_SGD(X_train, y_train, params, score, n_folds):
    """

    :param X_train:
    :param y_train:
    :param params:
    :param score:
    :param n_folds:
    :return: fitted classifier
    """
    # simple cross-validation over hyper parameters of classifier
    model = GridSearchCV(SGDClassifier(), param_grid=params, cv=n_folds, scoring='%s_micro' % score)
    model.fit(X_train, y_train)
    return model



if __name__ == '__main__':
    # load melt pool frame features
    X = np.loadtxt('/home/mahdi/Desktop/PhD_application/eth_additive_phd/interview/outputs/question_3/features_layer0002.csv')
    # load XYPV data
    _y = np.loadtxt('/home/mahdi/Desktop/PhD_application/eth_additive_phd/interview/outputs/question_2/XYPV_layer0002.csv')
    # digitize XYPV to get labels
    bin_num = 10
    bins_location_X = np.linspace(_y[:, 0].min(), _y[:, 0].max(), bin_num + 1)
    y_location_X = np.digitize(_y[:, 0], bins_location_X)
    bins_location_Y = np.linspace(_y[:, 1].min(), _y[:, 1].max(), bin_num + 1)
    y_location_Y = np.digitize(_y[:, 1], bins_location_Y)
    bins_power = np.linspace(_y[:, 2].min(), _y[:, 2].max(), bin_num + 1)
    y_power = np.digitize(_y[:, 2], bins_power)
    bins_velocity = np.linspace(_y[:,3].min(), _y[:,3].max(), bin_num+1)
    y_velocity = np.digitize(_y[:,3], bins_velocity)

    X_train, X_test, y_velocity_train, y_velocity_test = train_test_split(X, y_velocity, test_size=0.25)
    # to demonstrate cross validation we only consider one hyper parameter: regularization factor
    params = {'alpha': [1e-4, 1e-2, 1e0]}
    # cross validation score criteria and number of folds
    score = 'f1'
    n_folds = 5

    trained_model = model_SGD(X_train, y_velocity_train, params, score, n_folds)
    y_velocity_pred_train = trained_model.predict(X_train)
    print('Trained classifier train accuracy score is %{:.2f}'.format(accuracy_score(y_velocity_train, y_velocity_pred_train)*100))
    y_velocity_pred_test = trained_model.predict(X_test)
    print('Trained classifier test accuracy score is %{:.2f}'.format(accuracy_score(y_velocity_test, y_velocity_pred_test)*100))
