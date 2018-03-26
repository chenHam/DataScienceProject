import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import scikitplot as skplt
from pyspark.mllib.regression import LabeledPoint
from sklearn import preprocessing, model_selection, tree, naive_bayes
from pyspark import SparkContext, Row
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import expr
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import to_timestamp
from sklearn.metrics import confusion_matrix
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes
import hashlib
import pandasql as pdsql

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set_style('whitegrid')


def createHotelsDataChangedFile():
    # read csv file
    df = pd.read_csv('hotels_data.csv')

    # convert columns from string to datetime
    df['Snapshot Date'] = pd.to_datetime(df['Snapshot Date'])
    df['Checkin Date'] = pd.to_datetime(df['Checkin Date'])

    # calculate days difference beteween snapshot and checkin dates
    df['DayDiff'] = abs((df['Snapshot Date'] - df['Checkin Date']).dt.days)

    # extracts day name for each checkin date
    df['WeekDay'] = df['Checkin Date'].dt.strftime('%a')

    # calculate discount difference
    df['DiscountDiff'] = df['Original Price'] - df['Discount Price']

    # calculate discount percentage
    df['DiscountPerc'] = (df['DiscountDiff'] / df['Original Price']) * 100

    df['DayDiff'] = df['DayDiff'].astype(np.float64)
    df['Original Price'] = df['Original Price'].astype(np.float64)
    df['Discount Price'] = df['Discount Price'].astype(np.float64)

    # save to dataframe to a new csv file
    df.to_csv('Hotels_data_Changed.csv', index=False)

def createFormattedFileEx2():
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
    query = 'select SnapshotDate, CheckinDate, DiscountCode, HotelName, DayDiff, WeekDay, max(DiscountDiff) as DiscountDiff ' \
            'from cf group by SnapshotDate, CheckinDate, HotelName, DayDiff, WeekDay'

    df = pdsql.sqldf(query)

    # save to dataframe to a new csv file
    df.to_csv('formatted_data_ex2.csv', index=False)


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

def sklearn_classifiers():
    """
    Invoke Decision-Tree and Naive-Bayes classifiers (Using sklearn lib)
    :return: None
    :rtype: None
    """
    # data = pd.read_csv('formatted_removed_cols.csv')
    data = pd.read_csv('formatted_data_ex2.csv')
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])
    x = data[['SnapshotDate', 'CheckinDate', 'DayDiff', 'HotelName', 'WeekDay']]
    y = data['DiscountCode']
    nb_classifier(x, y)
    dt_classifier(x, y)


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
        matrix = pd.DataFrame(confusion_matrix(y_test, y_prediction))
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.values.sum() - (FP + FN + TP)
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        FNR = FN/(TP+FN)
        print(str(type(classifier)) + ": Accuracy is " + str(sklearn.metrics.accuracy_score(y_test, y_prediction) * 100) + \
            ", Precision is " + str(sklearn.metrics.precision_score(y_test, y_prediction, average='macro') * 100))
        print("False Positive:")
        i = 1
        for rate in FPR:
            print("Label", i, "=", rate)
            i += 1
        i = 1
        print("False Negative:")
        for rate in FNR:
            print("Label", i, "=", rate)
            i += 1
        i = 1
        print("True Positive:")
        for rate in TPR:
            print("Label", i, "=", rate)
            i += 1
        i = 1
        print("True Negative:")
        for rate in TNR:
            print("Label", i, "=", rate)
            i += 1
        # TPR = TP/(TP+FN)
        # print("True positive rate:")
        # print(TPR)
        # TNR = TN/(TN+FP)
        # print("True negative rate:")
        # print(TNR)
        # FPR = FP/(FP+TN)
        # print("False positive rate:")
        # print(FPR)
        # FNR = FN/(TP+FN)
        # print("False negative rate:")
        # print(FNR)
        # skplt.metrics.plot_roc_curve(y_test, y_prediction2)
        # plt.show()
    except Exception as e:
        print(e)


def nb_classifier(x, y):
    """
    Invoke 'classify' function with Naive-Bayes as model
    :param x: Features
    :param y: Labels
    :return: None
    :rtype: None
    """
    classify(naive_bayes.GaussianNB(), x, y)


def dt_classifier(x, y):
    """
    Invoke 'classify' function with Decision-Tree as model
    :param x: Features
    :param y: Labels
    :return: None
    :rtype: None
    """
    classify(sklearn.tree.DecisionTreeClassifier(criterion="entropy"), x, y)


def hotelname_to_float(name):
    """
    User defined function for hashing hotel name
    :param name: Hotel name
    :type name: str
    :return: Float hashing representation of the name
    :rtype: float
    """
    str_name = str(name.encode("utf-8"))
    return float(int(hashlib.md5(str_name.encode('utf-8')).hexdigest()[:16], 16))


def weekday_to_float(weekday):
    """
    User defined function for converting weekday to float
    :param weekday:
    :type weekday: str
    :return: Float representation of weekday
    :rtype: float
    """
    weekdays = {"Sun": 1, "Mon": 2, "Tue": 3, "Wed": 4, "Thu": 5, "Fri": 6, "Sat": 7}
    return float(weekdays.get(weekday))


def datetime_to_float(dts):
    """
    User defined function for converting date to float
    :param dts: Date string
    :return: Seconds since epoch
    :rtype: float
    """
    epoch = datetime.utcfromtimestamp(0)
    #dt = datetime.strptime(dts, '%m/%d/%Y')
    return (dts - epoch).total_seconds()


def spark_classifiers():
    """
    Activate Spark classifiers
    :return: None
    :rtype: None
    """
    # Start spark session
    spark = SparkSession.builder.master("local").appName("Classifier").getOrCreate()
    # Read CSV file
    # data = spark.read.format("csv") \
    #    .option("header", "true") \
    #    .option("inferSchema", "true") \
    #    .load("formatted_removed_cols.csv")
    data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("formatted_data_ex2.csv")
    data.cache()  # Cache data for faster reuse
    udf_hotel_name_to_float = udf(hotelname_to_float, DoubleType())  # User defined function for hashing hotel name
    udf_weekday_to_float = udf(weekday_to_float, DoubleType())  # User defined function for converting weekday to float
    udf_date_to_float = udf(datetime_to_float, DoubleType())  # User defined function for converting date to float
    # Pre-process data
    data = data.withColumn("DiscountCode", data["DiscountCode"]) \
        .withColumn("SnapshotDate", udf_date_to_float("SnapshotDate")) \
        .withColumn("CheckinDate", udf_date_to_float("CheckinDate")) \
        .withColumn("HotelName", udf_hotel_name_to_float("HotelName")) \
        .withColumn("WeekDay", udf_weekday_to_float("WeekDay"))
    # Select only relevant columns
    data = data.select("DiscountCode", "SnapshotDate", "CheckinDate", "HotelName", "DayDiff", "WeekDay")
    # Convert Data-Frame to RDD
    data = data.rdd.map(lambda x: LabeledPoint(x[0], x[1:]))
    # Split data randomly to test and training data
    trainingData, testData = data.randomSplit([0.7, 0.3])
    # Activate Spark Decision-Tree on the data
    spark_dt_classifier(trainingData, testData)
    # Activate Spark Naive-Bayes on the data
    spark_nb_classifier(trainingData, testData)


def spark_dt_classifier(training_data, test_data):
    """
    Activate Spark Decision-Tree on the data
    :param training_data: Training data to use
    :type training_data: PipelinedRDD
    :param test_data: Test data to use
    :type test_data: PipelinedRDD
    :return: None
    :rtype: None
    """
    model = DecisionTree.trainClassifier(training_data, numClasses=5, categoricalFeaturesInfo={},
                                         impurity='entropy', maxDepth=5, maxBins=32)
    predictions = model.predict(test_data.map(lambda x: x.features))
    labels_and_predictions = test_data.map(lambda lp: lp.label).zip(predictions)
    test_err = labels_and_predictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
    print('Test Error = ' + str(test_err))
    print('Learned classification tree model:')
    print(model.toDebugString())


def spark_nb_classifier(training_data, test_data):
    """
    Activate Spark Decision-Tree on the data
    :param training_data: Training data to use
    :type training_data: PipelinedRDD
    :param test_data: Test data to use
    :type test_data: PipelinedRDD
    :return: None
    :rtype: None
    """
    model = NaiveBayes.train(training_data, 1.0)
    prediction_and_label = test_data.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * prediction_and_label.filter(lambda pl: pl[0] == pl[1]).count() / test_data.count()
    print('model accuracy {}'.format(accuracy))




# create hotels_data_changed.csv file (ex1)
#createHotelsDataChangedFile()

# createHotelsDataChangedFile()
# createFormattedFileEx2()

sklearn_classifiers()
spark_classifiers()
