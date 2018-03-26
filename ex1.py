import pandas as pd
import numpy as np


def create_hotels_data_changed_file():
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


if __name__ == "__main__":
    create_hotels_data_changed_file()