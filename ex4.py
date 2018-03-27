import hashlib
from datetime import datetime
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.classification import NaiveBayes


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


#helper functions
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
    #dt = datetime.strptime(dts, '%m/%d/%Y')  # needed in python2.7 only
    return (dts - epoch).total_seconds()


if __name__ == "__main__":
    spark_classifiers()


