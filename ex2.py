import sklearn
import pandas as pd
import pandasql as pdsql
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, tree, naive_bayes


def create_formatted_file_ex2():
    # read csv file
    df = pd.read_csv('Hotels_data_Changed.csv')

    # remove whitespace from colnames
    df.rename(columns={'Snapshot Date': 'SnapshotDate'}, inplace=True)
    df.rename(columns={'Checkin Date': 'CheckinDate'}, inplace=True)
    df.rename(columns={'Discount Code': 'DiscountCode'}, inplace=True)
    df.rename(columns={'Hotel Name': 'HotelName'}, inplace=True)
    needed_columns = ['SnapshotDate', 'CheckinDate', 'DiscountCode', 'HotelName', 'DayDiff', 'WeekDay', 'DiscountDiff']

    # create new dataframe with only the needed columns
    cf = df[needed_columns]

    # query for filtering the rows with the max DiscountDiff for
    # each SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay combination
    query = 'select SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay, max(DiscountDiff) ' \
            'from cf group by SnapshotDate, CheckinDate, HotelName, DayDiff, WeekDay'

    df = pdsql.sqldf(query)

    # save to dataframe to a new csv file
    df.to_csv('formatted_data_ex2.csv', index=False)

def sklearn_classifiers():
    """
    Invoke Decision-Tree and Naive-Bayes classifiers (Using sklearn lib)
    :return: None
    :rtype: None
    """
    data = pd.read_csv('formatted_data_ex2.csv')
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])
    x = data[['SnapshotDate', 'CheckinDate', 'DayDiff', 'HotelName', 'WeekDay']]
    y = data['DiscountCode']
    nb_classifier(x, y)
    dt_classifier(x, y)


def dt_classifier(x, y):
    """
    Invoke 'classify' function with Decision-Tree as model
    :param x: Features
    :param y: Labels
    :return: None
    :rtype: None
    """
    classify(sklearn.tree.DecisionTreeClassifier(criterion="entropy"), x, y)


def nb_classifier(x, y):
    """
    Invoke 'classify' function with Naive-Bayes as model
    :param x: Features
    :param y: Labels
    :return: None
    :rtype: None
    """
    classify(naive_bayes.GaussianNB(), x, y)


def classify(classifier, x, y):
    """
    Classify (using sklearn lib) based on given classifier
    :param classifier: sklearn classifier
    :param x: Features
    :param y: Labels
    :return: None
    :rtype: None
    """
    try:
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
        classifier.fit(x_train, y_train)
        y_prediction = classifier.predict(x_test)
        y_prediction2 = classifier.predict_proba(x_test)
        print(str(type(classifier)) + ": Accuracy is " + str(sklearn.metrics.accuracy_score(y_test, y_prediction) * 100) + \
            ", Precision is " + str(sklearn.metrics.precision_score(y_test, y_prediction, average='macro') * 100))
        skplt.metrics.plot_roc_curve(y_test, y_prediction2)
        plt.show()
    except Exception as e:
        print(e)


def main():
    create_formatted_file_ex2()
    sklearn_classifiers()

if __name__ == '__main__':
    main()

