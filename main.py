import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
import sklearn.exceptions
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn import preprocessing

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set_style('whitegrid')


def main():
    # read csv file
    df = pd.read_csv('hotels_data.csv')

    # convert columns to datetime from string
    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'])
    df['Checkin Date'] = pd.to_datetime(df['Checkin Date'])

    # df['DayDiff'] = diff_dates(df['Snapshot Date'], df['Checkin Date'])

    # calculate days difference beteween snapshot and checkin dates
    df['DayDiff'] = abs((df['Snapshot Date'] - df['Checkin Date']).dt.days)

    # extracts day name for each checkin date
    df['WeekDay'] = df['Checkin Date'].dt.strftime('%a')

    df['DiscountDiff'] = df['Original Price'] - df['Discount Price']

    df['DiscountPerc'] = (df['DiscountDiff'] / df['Original Price']) * 100

    df['DayDiff'] = df['DayDiff'].astype(np.float64)
    df['Original Price'] = df['Original Price'].astype(np.float64)
    df['Discount Price'] = df['Discount Price'].astype(np.float64)

    df.to_csv('new_file.csv', sep=',')
    # createFormattedDataFile()

    # classificationPrediction(df)

    print(df)


# def classificationPrediction(df):
#     X = df[['Snapshot Date', 'Checkin Date', 'DayDiff']].as_matrix()
#     # X = df[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']].as_matrix()
#     # X = df[['DayDiff', 'Hotel Name', 'WeekDay']].as_matrix()
#     # X = df[['DayDiff', 'Original Price', 'Discount Price']].as_matrix()
#
#     Y = df['Discount Code'].as_matrix()
#
#     # Grab data
#     hotel_data = DataFrame(X, columns=['Snapshot Date', 'Checkin Date', 'DayDiff'])
#     # hotel_data = DataFrame(X, columns=['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay'])
#     # hotel_data = DataFrame(X, columns=['DayDiff', 'Hotel Name', 'WeekDay'])
#     # hotel_data = DataFrame(X, columns=['DayDiff', 'Original Price', 'Discount Price'])
#
#     # Grab Target
#     hotel_target = DataFrame(Y, columns=['Discount Code'])
#
#     hotel_target['Discount Code'] = hotel_target['Discount Code'].apply(categoryName)
#
#     # Create a combined Iris DataSet
#     hotel = pd.concat([hotel_data, hotel_target], axis=1)
#     # nbClassifier(hotel)
#
#     # Create a Logistic Regression Class object
#     # logreg = LogisticRegression()
#
#     # Split the data into Trainging and Testing sets
#     # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=3)
#
#     # Train the model with the training set
#     # logreg.fit(X_train, Y_train)
#
#     # Prediction from X_test
#     # Y_pred = logreg.predict(X_test)
#
#     # Check accuracy
#     # print(metrics.accuracy_score(Y_test, Y_pred))
#
#     # print(hotel)


def categoryName(num):
    ''' Takes in numerical class, returns flower name'''
    if num == 0:
        return 'First'
    elif num == 1:
        return 'Second'
    # elif num == 2:
    #    return 'Third'
    elif num == 3:
        return 'Fourth'
    else:
        return 'Else'


def createFormattedDataFile():
    df = pd.read_csv('new_file.csv')
    groups = df.groupby(['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay'], sort=False)
    # max = groups['DiscountPerc'].max()
    # temp = groups.transform(max) == df['DiscountPerc']
    maxRows = df.loc[groups["DiscountPerc"].idxmax()]

    maxRows.to_csv('formatted_max_file.csv', sep=',')
    print(maxRows)

    return


def nbClassifier():
    try:
        data = pd.read_csv('formatted_removed_cols.csv')
        for column in data.columns:
            if data[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data[column] = le.fit_transform(data[column])
        x = data[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']]
        y = data['Discount Code']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        y_prediction = classifier.predict(x_test)
        print "Naive-Bayes: Accuracy is " + str(accuracy_score(y_test, y_prediction) * 100) + \
            ", Precision is" + str(precision_score(y_test, y_prediction, average='macro') * 100) +  \
            ", ROC is" + str(roc_auc_score(y_test, y_prediction))  # TODO - ROC is multiclass format is not supported
    except UndefinedMetricWarning as e:
        # Do nothing
        pass
    except DeprecationWarning as e:
        # Do nothing
        pass
    except Exception as e:
        print(e)


def dtClassifier():
    try:
        data = pd.read_csv('formatted_removed_cols.csv')
        for column in data.columns:
            if data[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data[column] = le.fit_transform(data[column])
        x = data[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']]
        y = data['Discount Code']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        classifier = DecisionTreeClassifier(criterion="entropy")
        classifier.fit(x_train, y_train)
        y_prediction = classifier.predict(x_test)
        print "Decision-Tree: Accuracy is " + str(accuracy_score(y_test, y_prediction) * 100) + \
            ", Precision is" + str(precision_score(y_test, y_prediction, average='macro') * 100) +  \
            ", ROC is" + str(roc_auc_score(y_test, y_prediction))  # TODO - ROC is multiclass format is not supported
    except UndefinedMetricWarning as e:
        # Do nothing
        pass
    except DeprecationWarning as e:
        # Do nothing
        pass
    except Exception as e:
        print(e)


# main()
dtClassifier()
nbClassifier()
