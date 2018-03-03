import warnings

from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from pyspark.mllib.regression import LabeledPoint
from sklearn import preprocessing, model_selection, tree, naive_bayes
from pyspark import SparkContext, Row
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import expr
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes
import hashlib

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


def sklearn_classifiers():
    nbClassifier()
    dtClassifier()


def classify(classifier):
    try:
        data = pd.read_csv('formatted_removed_cols.csv')
        for column in data.columns:
            if data[column].dtype == type(object):
                le = preprocessing.LabelEncoder()
                data[column] = le.fit_transform(data[column])
        x = data[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']]
        y = data['Discount Code']
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
        classifier.fit(x_train, y_train)
        y_prediction = classifier.predict(x_test)
        print str(type(classifier)) + ": Accuracy is " + str(sklearn.metrics.accuracy_score(y_test, y_prediction) * 100) + \
            ", Precision is " + str(sklearn.metrics.precision_score(y_test, y_prediction, average='macro') * 100)  # +  \
            # ", ROC is" + str(sklearn.metrics.roc_auc_score(y_test, y_prediction))  # TODO - ROC is multiclass format is not supported
    except Exception as e:
        print(e)


def nbClassifier():
    # try:
    #     data = pd.read_csv('formatted_removed_cols.csv')
    #     for column in data.columns:
    #         if data[column].dtype == type(object):
    #             le = preprocessing.LabelEncoder()
    #             data[column] = le.fit_transform(data[column])
    #     x = data[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']]
    #     y = data['Discount Code']
    #     x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
    #     classifier = naive_bayes.GaussianNB()
    #     classifier.fit(x_train, y_train)
    #     y_prediction = classifier.predict(x_test)
    #     print "Naive-Bayes: Accuracy is " + str(sklearn.metrics.accuracy_score(y_test, y_prediction) * 100) + \
    #         ", Precision is " + str(sklearn.metrics.precision_score(y_test, y_prediction, average='macro') * 100)  # +  \
    #         # ", ROC is" + str(sklearn.metrics.roc_auc_score(y_test, y_prediction))  # TODO - ROC is multiclass format is not supported
    # except Exception as e:
    #     print(e)
    classify(naive_bayes.GaussianNB())


def dtClassifier():
    # try:
    #     data = pd.read_csv('formatted_removed_cols.csv')
    #     for column in data.columns:
    #         if data[column].dtype == type(object):
    #             le = preprocessing.LabelEncoder()
    #             data[column] = le.fit_transform(data[column])
    #     x = data[['Snapshot Date', 'Checkin Date', 'DayDiff', 'Hotel Name', 'WeekDay']]
    #     y = data['Discount Code']
    #     x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
    #     classifier = tree.DecisionTreeClassifier(criterion="entropy")
    #     classifier.fit(x_train, y_train)
    #     y_prediction = classifier.predict(x_test)
    #     print "Decision-Tree: Accuracy is " + str(sklearn.metrics.accuracy_score(y_test, y_prediction) * 100) + \
    #         ", Precision is " + str(sklearn.metrics.precision_score(y_test, y_prediction, average='macro') * 100)  # +  \
    #         # ", ROC is" + str(sklearn.metrics.roc_auc_score(y_test, y_prediction))  # TODO - ROC is multiclass format is not supported
    # except Exception as e:
    #     print(e)
    classify(sklearn.tree.DecisionTreeClassifier(criterion="entropy"))


def hotelname_to_int(name):
    str_name = str(name.encode("utf-8"))
    return float(int(hashlib.md5(str_name).hexdigest()[:16], 16))


def weekday_to_int(weekday):
    weekdays = {"Sun": 1, "Mon": 2, "Tue": 3, "Wed": 4, "Thu": 5, "Fri": 6, "Sat": 7}
    return float(weekdays.get(weekday))


def datetime_to_int(dts):
    epoch = datetime.utcfromtimestamp(0)
    dt = datetime.strptime(dts, '%m/%d/%Y')
    return (dt - epoch).total_seconds()


def spark_classifiers():
    spark = SparkSession.builder.master("local").appName("Classifier").getOrCreate()
    data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("formatted_removed_cols.csv")
    data.cache()  # Cache data for faster reuse
    udfHotelNameToInt = udf(hotelname_to_int, DoubleType())
    udfWeekDayToInt = udf(weekday_to_int, DoubleType())
    udfDateToInt = udf(datetime_to_int, DoubleType())
    data = data.withColumn("DiscountCode", data["Discount Code"]) \
        .withColumn("SnapshotDate", udfDateToInt("Snapshot Date")) \
        .withColumn("CheckinDate", udfDateToInt("Checkin Date")) \
        .withColumn("HotelName", udfHotelNameToInt("Hotel Name")) \
        .withColumn("WeekDay", udfWeekDayToInt("WeekDay"))
    data = data.select("DiscountCode", "SnapshotDate", "CheckinDate", "Days", "HotelName", "DayDiff", "WeekDay")
    data = data.rdd.map(lambda x: LabeledPoint(x[0], x[1:]))
    trainingData, testData = data.randomSplit([0.7, 0.3])
    spark_dt_classifier(trainingData, testData)
    spark_nb_classifier(trainingData, testData)


def spark_dt_classifier(trainingData, testData):
    # spark = SparkSession.builder.master("local").appName("Classifier").getOrCreate()
    # data = spark.read.format("csv") \
    #     .option("header", "true") \
    #     .option("inferSchema", "true") \
    #     .load("formatted_removed_cols.csv")
    # data.cache()  # Cache data for faster reuse
    # data = data.withColumn("DiscountCode", data["Discount Code"].cast(DoubleType())) \
    #     .withColumn("SnapshotDate", data["Snapshot Date"].cast(DoubleType())) \
    #     .withColumn("CheckinDate", data["Checkin Date"].cast(DoubleType())) \
    #     .withColumn("Days", data["Days"].cast(DoubleType())) \
    #     .withColumn("HotelName", data["Hotel Name"].cast(DoubleType())) \
    #     .withColumn("DayDiff", data["DayDiff"].cast(DoubleType())) \
    #     .withColumn("WeekDay", data["WeekDay"].cast(DoubleType()))
    # data = data.select("DiscountCode", "SnapshotDate", "CheckinDate",
    #                    "Days", "HotelName", "DayDiff", "WeekDay")
    # data = data.rdd.map(lambda x: LabeledPoint(x[0], x[1:]))
    # trainingData, testData = data.randomSplit([0.7, 0.3])
    model = DecisionTree.trainClassifier(trainingData, numClasses=5, categoricalFeaturesInfo={},
                                         impurity='entropy', maxDepth=5, maxBins=32)
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())
    # spark = SparkSession.builder.master("local").appName("Classifier").getOrCreate()
    # data = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED") \
    #     .load("formatted_removed_cols.csv")
    # print type(data)


def spark_nb_classifier(trainingData, testData):
    # trainingData, testData = data.randomSplit([0.7, 0.3])
    model = NaiveBayes.train(trainingData, 1.0)
    predictionAndLabel = testData.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / testData.count()
    print('model accuracy {}'.format(accuracy))


# main()
# sklearn_classifiers()
spark_classifiers()
