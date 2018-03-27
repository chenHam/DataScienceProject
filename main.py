import ex1
import ex2
import ex3
import ex4
import os

# Needed for spark activation
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home"
os.environ['PYSPARK_DRIVER_PYTHON'] = "/Users/yschori/anaconda/bin/python3.5"
os.environ['PYSPARK_PYTHON'] = "/Users/yschori/anaconda/bin/python3.5"


def main():
    ex1.create_hotels_data_changed_file()
    ex2.main()
    ex3.main()
    ex4.spark_classifiers()


if __name__ == '__main__':
    main()