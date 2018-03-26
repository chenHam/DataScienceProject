import pandas as pd
import pandasql as pdsql
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt


pysql = lambda q: pdsql.sqldf(q, globals())

def top_150_hotels(df):
    top_hotels_query = 'select HotelName, count(HotelName) ' \
                     'from df ' \
                     'group by HotelName ' \
                     'order by count(HotelName) desc ' \
                     'limit 150'

    return pdsql.sqldf(top_hotels_query)

def top_40_checkin(df):
    # 40 check in dates most records
    top_checkin_query = 'select CheckinDate, count(CheckinDate) ' \
                      'from df ' \
                      'group by CheckinDate ' \
                      'order by count(CheckinDate) desc ' \
                      'limit 40'

    return pdsql.sqldf(top_checkin_query)

def get_main_hotels_data(df, top_hotels_df, top_checkin_df):
    # DROP THE UNRELEVANT DATA FROM THE MAIN DATA FRAME
    main_data_query = 'select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) ' \
                    'from df ' \
                    'where HotelName in (select HotelName from top_hotels_df) ' \
                    'and CheckinDate in (select CheckinDate from top_checkin_df) ' \
                    'group by HotelName, CheckinDate, DiscountCode'
    names = {'HotelName': 'HotelName', 'CheckinDate': 'CheckinDate', 'DiscountCode': 'DiscountCode',
             'min(DiscountPrice)': 'DiscountPrice'}

    return pdsql.sqldf(main_data_query).rename(columns=names)

def get_hotels_price_per_code_per_date(df, top_checkin_df, main_data_df):
    # create df containing all discount codes
    discount_code_df = pdsql.sqldf('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)')

    # cross join checkin_date with discount_code_df
    df_cross_checkin_discount_code = df_crossjoin(top_checkin_df[['CheckinDate']], discount_code_df[["DiscountCode"]])

    # create df with column datepluscode = "date"_"code"
    df_date_plus_code = df_cross_checkin_discount_code
    df_date_plus_code['datePlusCode'] = df_cross_checkin_discount_code['CheckinDate'] + '_' + df_cross_checkin_discount_code[
        'DiscountCode']

    # select hotel_name and price for each date and discount code combination
    discount_code_date_query = 'select a.HotelName, a.DiscountPrice, b.datePlusCode ' \
                                'from main_data_df as a ' \
                                'inner join df_date_plus_code as b ' \
                                'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
    discount_code_date_df = pdsql.sqldf(discount_code_date_query)
    discount_code_date_df['DiscountPrice'] = discount_code_date_df['DiscountPrice'].astype('int')

    return discount_code_date_df

def normalize_data(discount_code_date_df):
    # Normalize
    minPrice = pdsql.sqldf('select min(DiscountPrice) from discount_code_date_df')['min(DiscountPrice)'][0]
    maxPrice = pdsql.sqldf('select max(DiscountPrice) from discount_code_date_df')['max(DiscountPrice)'][0]
    discount_code_date_df['DiscountPrice'] = (
                (discount_code_date_df['DiscountPrice'] - minPrice) / (maxPrice - minPrice) * 100)

    return discount_code_date_df

def cluster_hotels_by_price(discount_code_date_df):
    finalDF = discount_code_date_df.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
    finalDF.fillna(value=-1, inplace=True)

    finalDF.to_csv('pivot.csv')

    finalDF.drop(finalDF.index[0], inplace=True)

    # Calculate the distance between each sample
    Z = hierarchy.linkage(finalDF, 'ward')
    # Plot with Custom leaves
    hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=5, labels=finalDF.index)
    plt.show()


# helper function
def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    return res

def main():
    colnames = ['SnapshotID', 'SnapshotDate', 'CheckinDate', 'Days', 'OriginalPrice', 'DiscountPrice', 'DiscountCode',
                'AvailableRooms',
                'HotelName', 'HotelStars']

    # read the basic hotels_data from csv
    df = pd.read_csv('hotels_data.csv', names=colnames, header=None, low_memory=False)

    # get top frequent 150 hotels
    top_hotels_df = top_150_hotels(df)

    # get top frequent 40 check in dates
    top_checkin_df = top_40_checkin(df)

    # get hotels data for the hotels in top_hotels_df and top_checkin_df
    main_df = get_main_hotels_data(df, top_hotels_df, top_checkin_df)

    discount_code_date_df = get_hotels_price_per_code_per_date(df, top_checkin_df, main_df)

    discount_code_date_df = normalize_data(discount_code_date_df)

    print(discount_code_date_df)

    cluster_hotels_by_price(discount_code_date_df)


if __name__ == "__main__":
    main()
