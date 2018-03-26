import pandas as pd
import pandasql as pdsql
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt


def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    return res

colnames = ['SnapshotID', 'SnapshotDate', 'CheckinDate', 'Days', 'OriginalPrice', 'DiscountPrice', 'DiscountCode',
            'AvailableRooms',
            'HotelName', 'HotelStars']

df = pd.read_csv('hotels_data.csv', names=colnames, header=None, low_memory=False)
pysql = lambda q: pdsql.sqldf(q, globals())

# 150 hotels most records
topHotelsQuery = 'select HotelName, count(HotelName) ' \
     'from df ' \
     'group by HotelName ' \
     'order by count(HotelName) desc ' \
     'limit 150'

topHotelsDF = pysql(topHotelsQuery)

# 40 check in dates most records
topCheckInQuery = 'select CheckinDate, count(CheckinDate) ' \
     'from df ' \
     'group by CheckinDate ' \
     'order by count(CheckinDate) desc ' \
     'limit 40'

topCheckInDF = pysql(topCheckInQuery)

# DROP THE UNRELEVANT DATA FROM THE MAIN DATA FRAME
mainDataQuery = 'select HotelName, CheckinDate,DiscountCode, min(DiscountPrice) ' \
       'from df ' \
       'where HotelName in (select HotelName from topHotelsDF) ' \
       'and CheckinDate in (select CheckinDate from topCheckInDF) ' \
       'group by HotelName, CheckinDate, DiscountCode'
names = {'HotelName': 'HotelName', 'CheckinDate': 'CheckinDate', 'DiscountCode': 'DiscountCode',
         'min(DiscountPrice)': 'DiscountPrice'}
mainDataDF = pysql(mainDataQuery).rename(columns=names)

# Query 3.1
discountCodeDF = pysql('select distinct DiscountCode from df where DiscountCode in (1,2,3,4)')

# Query 3.2
dfCrossCheckInDiscountCode = df_crossjoin(topCheckInDF[['CheckinDate']], discountCodeDF[["DiscountCode"]])

# Query 3.3
dfDatePlusCode = dfCrossCheckInDiscountCode
dfDatePlusCode['datePlusCode'] = dfCrossCheckInDiscountCode['CheckinDate'] + '_' + dfCrossCheckInDiscountCode['DiscountCode']

# Query 3.4 - select price for each hotel in each discount price and date
priceByHotelDateCodeQuery = 'select a.HotelName, a.DiscountPrice, b.datePlusCode ' \
      'from mainDataDF as a ' \
      'inner join dfDatePlusCode as b ' \
      'on a.DiscountCode=b.DiscountCode and a.CheckinDate=b.CheckinDate '
priceByHotelDateCodeDF = pysql(priceByHotelDateCodeQuery)
priceByHotelDateCodeDF['DiscountPrice'] = priceByHotelDateCodeDF['DiscountPrice'].astype('int')

#Normalize
minPrice = pysql('select min(DiscountPrice) from priceByHotelDateCodeDF')['min(DiscountPrice)'][0]
maxPrice = pysql('select max(DiscountPrice) from priceByHotelDateCodeDF')['max(DiscountPrice)'][0]
priceByHotelDateCodeDF['DiscountPrice'] = ((priceByHotelDateCodeDF['DiscountPrice'] - minPrice) / (maxPrice - minPrice) * 100)


#Query 4
finalDF = priceByHotelDateCodeDF.pivot(index='HotelName', columns='datePlusCode', values='DiscountPrice')
finalDF.fillna(value=-1, inplace=True)

finalDF.to_csv('pivot.csv')

finalDF.drop(finalDF.index[0], inplace=True)


# Calculate the distance between each sample
Z = hierarchy.linkage(finalDF, 'ward')
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=5, labels=finalDF.index)
plt.show()