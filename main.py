from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier,MultilayerPerceptronClassifier\
                                      ,LogisticRegression,LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

import io
import s3fs
import time
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


main_start_time = time.time()
# S3_BUCKET_READ_DATA_PATH = 's3://myp1/final.csv'
# S3_OUTPUT_PATH = 's3://myp1/output/'

class CreateSession:
    '''
    Returns a Spark Session
    '''
    def create_session(self):
        sc = SparkSession.builder\
            .master("local[*]")\
            .appName("Air Line Delay Prediction") \
            .getOrCreate()
        return sc



class ImportData:
    '''
    Reads a data from specified path and returns a RDD
    '''
    def read_csv(self,sc,filepath):
        df = sc.read\
            .option("inferschema",True)\
            .option("header",True)\
            .csv(filepath)
        print("Total Number of Rows  are {}".format(df.count()))
        print("Total Number of Columns are {}".format(len(df.columns)))
        print("\n")
        return df


class FeatureEngineering:
    '''
    Basic Feature Engineering methods available
    '''
    def drop_columns(self,df,columns):
        """
        :param df: Pass DataFrame whose Columns Needs to be Dropped
        :param columns: Pass a List of Columns
        :return: None
        """
        print("Total No of Columns Before Dropping unimportant columns:",len(df.columns))
        new_df = df.drop(*columns)
        print("Total of Columns After Dropping unimportant columns:", len(new_df.columns))
        print("\n")
        return new_df

    def count_null(self,df):
        """
        :param df: Pass DataFrame whose Null values has to be counted
        :return: None
        """
        #cached_df = df.cache() --> Do this in AWS
        new_df = df.select([(((count(when(col(c).isNull(),c)))/df.count())*100).alias(c) for c in df.columns])
        new_df.show()
        print("\n")


    def extract_year_month(self,df):
        """
        :param df: Pass DataFrame whose Year, Month, Day and DayName has to be extarcted
        :return: New DataFrame with Year, Month, Day and DayName
        """
        new_df = df.withColumn("Year",year(df.FL_DATE))\
            .withColumn('Month',month(df.FL_DATE))\
            .withColumn('Day',dayofmonth(df.FL_DATE))\
            .withColumn("Day_Name",date_format(col("FL_DATE"), "EEEE"))\
            .withColumn("Day_OfWeek", dayofweek(df.FL_DATE))
        new_df.show(5)
        print("\n")
        return new_df


    def combine_all_dataframes(self,*dfs):
        """
        :param:     Pass a List of All DataFrames which has to be merged
        :return:    Merged DataFrame
        """
        print("Combining all DataFrames..")
        combined = reduce(DataFrame.unionByName,dfs)
        print("DataFrames Combined")
        return combined


    def map_state_city(self,df1,df2):
       """
       :param df1: Requires DataFrame whose State and City has to be Found
       :param df2: Requires Reference DataFrame of Airport Codes,State,City
       :return: Returns New DataFrame with OriginState, OriginCity, DestinationState, DestiantionCity
       """
       new_df = df1.join(df2,df1.ORIGIN == df2.IATA,"left").drop("Airport","IATA")\
            .withColumnRenamed("State","OriginState").withColumnRenamed("City","OriginCity")\
            .join(df2,df1.DEST == df2.IATA,"left").drop("Airport","IATA")\
            .withColumnRenamed("State","DestinationState").withColumnRenamed("City","DestinationCity")
       new_df.show(5)
       print("\n")
       return new_df


    def seasons(self,sc,df):
        df.createOrReplaceTempView("temp_df")
        new_df = sc.sql("SELECT *, \
                                CASE \
                                WHEN Month IN (3,4,5)   THEN  'SPRING'  \
                                WHEN Month IN (6,7,8)   THEN  'SUMMER'  \
                                WHEN Month IN (9,10,11) THEN  'AUTUMN'  \
                                WHEN Month IN (12,1,2)  THEN  'WINTER'  \
                                END AS Seasons \
                        FROM temp_df")
        new_df.show(5)
        print("\n")
        return new_df


    def parts_of_day(self,sc,df):
        df.createOrReplaceTempView("temp_df")
        new_df = sc.sql("SELECT *, \
                                CASE \
                                WHEN CRS_DEP_TIME >= 0000 AND CRS_DEP_TIME < 400   THEN  'Early Morning'  \
                                WHEN CRS_DEP_TIME >= 400 AND CRS_DEP_TIME < 1200   THEN  'Morning'  \
                                WHEN CRS_DEP_TIME >= 1200 AND CRS_DEP_TIME < 1800   THEN  'Afternoon'  \
                                WHEN CRS_DEP_TIME >= 1800 AND CRS_DEP_TIME <= 2359   THEN  'Evening/Night'  \
                                END AS Parts_of_day \
                        FROM temp_df")
        new_df.show(5)
        print("\n")
        return new_df


    def cancellationcode_without_1(self,df):
        print("Before Dropping Cancellation Code 1 total rows:",df.count())
        new_df = df.filter(df["CANCELLED"] != 1)
        print("After Dropping Cacellation Code 1, total rows:",new_df.count())
        return new_df


    def remove_outliers(self,df):
        print("Before Dropping Outliers, total rows:",df.count())
        new_df = df.filter(df["ARR_DELAY"] < 400).filter(df["DEP_DELAY"] < 400)
        print("After Dropping Outliers, total rows:",new_df.count())
        return new_df



class DataVisualization:

    def plot_most_desired_departure_location(self,sc,df):
        df.createOrReplaceTempView("most_desired_departure_location")
        new_df = sc.sql(
            "SELECT ORIGIN,COUNT(ORIGIN) as Number "
            "FROM most_desired_departure_location "
            "GROUP BY ORIGIN "
            "ORDER BY Number desc LIMIT 15")
        print("Plotting Top 15 Most Desired Departure Location")
        plt.figure(figsize=(30, 10))
        plt.bar(new_df.select(collect_list('ORIGIN')).first()[0], new_df.select(collect_list('Number')).first()[0])
        plt.title("15 Most Desired Departure Location")
        plt.xlabel("Departure Location")
        plt.ylabel("Total Number of Flights")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + "Plotting Top 15 Most Desired Departure Location", 'wb') as f:
            f.write(img_data.getbuffer())

    def plot_no_flights_per_month(self,sc,df,year):
        df.createOrReplaceTempView("flights_per_month")
        new_df = sc.sql("SELECT Month,COUNT(*) as Number FROM flights_per_month GROUP BY Month ORDER BY Month")
        fig = plt.figure(figsize=(20, 10))
        plt.bar(new_df.select(collect_list('Month')).first()[0],new_df.select(collect_list('Number')).first()[0])
        plt.title("Number of Flights Each Month for Year {}".format(year))
        plt.xlabel("Month")
        plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['January','Febuary','March','April','May','June','July','August','September',\
                                                  'October','November','December'])
        plt.ylabel("Total Number of Flights")
        # plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH+"Number of Flights Per Month for {}.png".format(year), 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+"Number of Flights Per Month for {}.png".format(year), dpi=fig.dpi)


    def plot_no_flights_by_each_carrier(self,sc,df,year):
        df.createOrReplaceTempView("flights_per_carrier")
        new_df = sc.sql("SELECT OP_CARRIER,COUNT(*) as Number FROM flights_per_carrier GROUP BY OP_CARRIER ORDER BY Number")
        fig = plt.figure(figsize=(20, 10))
        plt.bar(new_df.select(collect_list('OP_CARRIER')).first()[0], new_df.select(collect_list('Number')).first()[0])
        plt.title("Number of Flights Flown by Each Carrier in Year {}".format(year))
        plt.xlabel("Carrier Names")
        plt.ylabel("Total Number of Flights Flown")
        # plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Number of Flights By Each Carrier in {}.png'.format(year), 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Number of Flights By Each Carrier in {}.png'.format(year), dpi=fig.dpi)

    def plot_top_10_most_desired_destination_states(self,sc,df):
        df.createOrReplaceTempView("most_desired_destination_state")
        new_df = sc.sql("SELECT DestinationState,COUNT(DEP_TIME) as Number FROM most_desired_destination_state GROUP BY DestinationState ORDER BY Number desc LIMIT 10")
        fig = plt.figure(figsize=(20, 10))
        plt.bar(new_df.select(collect_list('DestinationState')).first()[0], new_df.select(collect_list('Number')).first()[0])
        plt.title("10 Most Desired Destination States ")
        plt.xlabel("Destination States")
        plt.ylabel("Total Number of Flights Flown")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 10 Desired Destination States', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 10 Desired Destination States in {}.png'.format(year), dpi=fig.dpi)

    def plot_top_5_most_desired_destination_states_each_year(self, sc, df):
        df.createOrReplaceTempView("top_5_most_desired_destination_state_year")
        temp_df = sc.sql("SELECT Year,DestinationState,COUNT(DEP_TIME) as Number from top_5_most_desired_destination_state_year \
                          GROUP BY Year,DestinationState ORDER BY Year,Number desc")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql(
            "WITH top_5_destination_state_year as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY Number desc) as rnk \
                FROM temp_df_sql ) \
            SELECT Year,DestinationState, Number FROM top_5_destination_state_year WHERE rnk <= 5 \
             GROUP BY Year,DestinationState,Number ORDER BY Year,Number desc")
        pd_new_df = new_df.toPandas()
        print("Plotting Top 5 Destination States for Each Year")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Top 5 Destination States for Each Year")
        sns.barplot(x=pd_new_df["Year"], y=pd_new_df["Number"], hue=pd_new_df["DestinationState"], data=pd_new_df)
        plt.xlabel("Years")
        plt.ylabel("Total Number of Flights Flown to Each Destination")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Destination States Each Year.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Destination States Each Year.png', dpi=fig.dpi)

    def plot_top_5_most_desired_destination_city_customstate_each_year(self, sc, df,state):
        """
        :param sc:  Pass SparkSession
        :param df:  Pass DataFrame
        :param state: Pass State
        :return: None
        """
        df.createOrReplaceTempView("top_5_most_desired_destination_city_customstate_each_year")
        temp_df = sc.sql("SELECT Year,DestinationCity,COUNT(DEP_TIME) as Number \
                            FROM top_5_most_desired_destination_city_customstate_each_year \
                            WHERE DestinationState = '{}' GROUP BY Year,DestinationCity ORDER BY Year,Number desc".format(state))
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql(
            "WITH top_5_destination_city_customstate_year as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY Number desc) as rnk \
                FROM temp_df_sql ) \
            SELECT Year,DestinationCity, Number FROM top_5_destination_city_customstate_year WHERE rnk <= 5 \
             GROUP BY Year,DestinationCity,Number ORDER BY Year,Number desc")
        pd_new_df = new_df.toPandas()
        title = "Plotting Top 5 Destination City for {} for Each Year".format(state)
        print(title)
        fig = plt.figure(figsize=(20, 10))
        plt.title(title)
        sns.barplot(x=pd_new_df["Year"], y=pd_new_df["Number"], hue=pd_new_df["DestinationCity"], data=pd_new_df)
        plt.xlabel("Years")
        plt.ylabel("Total Number of Flights Flown to Each Destination")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Desired City for Custom State Each Year.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Desired City for Custom State Each Year.png', dpi=fig.dpi)


    def plot_most_desired_destination_customstate1_customestate2_each_year(self, sc, df,state1,state2):

        df.createOrReplaceTempView("most_desired_destination_customstate1_customestate2_each_year")
        temp_df = sc.sql("WITH state1 AS \
                          ( SELECT Year,COUNT(DEP_TIME) as State1 FROM most_desired_destination_customstate1_customestate2_each_year \
                             WHERE DestinationState = '{0}' GROUP BY Year), \
                           state2 AS \
                           (SELECT Year,COUNT(DEP_TIME) as State2 FROM most_desired_destination_customstate1_customestate2_each_year \
                             WHERE DestinationState = '{1}' GROUP BY Year) \
                            SELECT s1.Year,State1,State2 \
                            FROM state1 s1 \
                            JOIN state2 s2 \
                            ON s1.Year = s2.Year \
                            ORDER BY s1.Year ".format(state1,state2))
        title = "Plotting Total Flights for Destination {0} and Destination {1} for Each Year".format(state1,state2)
        print(title)
        fig = plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.plot(temp_df.select(collect_list('Year')).first()[0],temp_df.select(collect_list('State1')).first()[0],'-r',label=state1)
        plt.plot(temp_df.select(collect_list('Year')).first()[0], temp_df.select(collect_list('State2')).first()[0],
                 '-b', label=state2)
        plt.legend(loc="upper left")
        plt.xlabel("Years")
        plt.ylabel("Total Number of Flights Flown")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Line Plot to compare total flighs for 2 states.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Line Plot to compare total flighs for 2 states.png', dpi=fig.dpi)



    def plot_most_desired_day(self,sc,df,year):
        df.createOrReplaceTempView("most_desired_day")
        new_df = sc.sql("SELECT Day_Name,COUNT(*) as Number FROM most_desired_day GROUP BY Day_Name ORDER BY Number desc")
        fig = plt.figure(figsize=(10, 10))
        plt.bar(new_df.select(collect_list('Day_Name')).first()[0], new_df.select(collect_list('Number')).first()[0])
        plt.title("Most Desired Day in Year {}".format(year))
        plt.xlabel("Days")
        plt.ylabel("Total Number of Flights Flown")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Most Desired Day in {}.png'.format(year), 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Most Desired Day in {}.png'.format(year), dpi=fig.dpi)

    def plot_total_flights_top_5_carriers_eachday(self,sc,df):
        df.createOrReplaceTempView("total_flights_top_5_carriers_eachday")
        temp_df = sc.sql("SELECT Day_Name,OP_CARRIER,Day_OfWeek,COUNT(DEP_TIME) as Number FROM total_flights_top_5_carriers_eachday \
                         GROUP BY Day_Name,OP_CARRIER,Day_OfWeek ORDER BY Day_OfWeek")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql(
            "WITH top_5_eachday as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Day_Name ORDER BY Number desc) as rnk \
                FROM temp_df_sql \
                ) \
            SELECT Day_Name, OP_CARRIER, Day_OfWeek, Number FROM top_5_eachday WHERE rnk <= 5 \
            GROUP BY Day_Name, OP_CARRIER, Day_OfWeek, Number ORDER BY Day_OfWeek,Number desc ")
        pd_new_df = new_df.toPandas()
        print("Plotting Total Flights Flown by Top 5 Carriers for Each Day")
        fig = plt.figure(figsize=(20, 10))
        sns.barplot(x=pd_new_df["Day_Name"], y=pd_new_df["Number"], hue=pd_new_df["OP_CARRIER"], data=pd_new_df)
        plt.title('Top 5 Carriers for Each Day')
        plt.xlabel("Days")
        plt.ylabel("Total Number of Flights Flown by Each Carrier for Each Day")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Carriers Each Day.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Carriers Each Day.png', dpi=fig.dpi)

    def plot_top_5_carriers(self, sc, df):
        df.createOrReplaceTempView("top_5_carriers")
        temp_df = sc.sql("SELECT Year,OP_CARRIER,COUNT(DEP_TIME) as Number from top_5_carriers GROUP BY Year,OP_CARRIER")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql(
            "WITH top_5 as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY Number desc) as rnk \
                FROM temp_df_sql \
                ) \
            SELECT Year,OP_CARRIER, Number FROM top_5 WHERE rnk <= 5 GROUP BY Year,OP_CARRIER,Number ORDER BY Year,Number desc")
        pd_new_df = new_df.toPandas()
        print("Plotting Top 5 Carriers for Each Year")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Top 5 Carriers for Each Year")
        sns.barplot(x=pd_new_df["Year"], y=pd_new_df["Number"], hue=pd_new_df["OP_CARRIER"], data=pd_new_df)
        plt.xlabel("Years")
        plt.ylabel("Total Number of Flights Flown by Each Carrier")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH +'Top 5 Carriers Each Year.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Carriers Each Year.png', dpi=fig.dpi)


    def plot_avg_departuredelay_arrivaldelay_top_5_carrier(self, sc, df):
        df.createOrReplaceTempView("avg_departuredelay_arrivaldealy_top_5_carrier")
        temp_df = sc.sql("SELECT OP_CARRIER,COUNT(DEP_TIME) as NumberOfFlightsFlown,AVG(DEP_DELAY) as AverageDepartureDelay, \
                           AVG(ARR_DELAY) as AverageArrivalDelay from avg_departuredelay_arrivaldealy_top_5_carrier \
                           GROUP BY OP_CARRIER ORDER BY NumberOfFlightsFlown desc LIMIT 5")

        pd_new_df = temp_df.toPandas()
        print("Plotting Average Departure Delay AND Average Arrival Delay Time of Top 5 Carriers")
        pd_new_df.plot(x="OP_CARRIER", y=["AverageDepartureDelay","AverageArrivalDelay"], kind='bar', figsize=(20, 10), \
                       title='Average Departure Delay AND Average Arrival Delay of Top 5 Carriers')

        plt.xlabel("Top 5 Carriers")
        plt.ylabel("Delay in Minutes")
        #plt.show()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH +'Top 5 Carriers, Average Arrival Delay and Average Departure Delay.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #plt.savefig(S3_OUTPUT_PATH+'Top 5 Carriers, Average Arrival Delay and Average Departure Delay.png')


    def plot_avg_departure_delaytime_top_5_airports(self, sc, df):
        df.createOrReplaceTempView("avg_departure_delaytime_top_5_airports")
        temp_df = sc.sql("SELECT Year,ORIGIN,COUNT(DEP_TIME) as NumberOfFlightsFlown,AVG(DEP_DELAY) as AverageDepartureDelay from avg_departure_delaytime_top_5_airports GROUP BY Year,ORIGIN")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql( \
            "WITH avg_departure_delaytime_top_5_airports as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY NumberOfFlightsFlown desc) as rnk \
                FROM temp_df_sql ) \
            SELECT Year,ORIGIN, AverageDepartureDelay FROM avg_departure_delaytime_top_5_airports WHERE rnk <= 5 GROUP BY Year,ORIGIN,AverageDepartureDelay ORDER BY Year,AverageDepartureDelay desc")
        pd_new_df = new_df.toPandas()
        print("Plotting Average Departure Delay for Top 5 Airports for Each Year")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Average Departur Delay of Top 5 Airports for Each Year")
        sns.barplot(x=pd_new_df["Year"], y=pd_new_df["AverageDepartureDelay"], hue=pd_new_df["ORIGIN"], data=pd_new_df)
        plt.xlabel("Years")
        plt.ylabel("Average Departure Delay of Top 5 Airports in Minutes")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH +'Top 5 Airports, Average Average Departure Delay Time.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Airports, Average Average Departure Delay Time.png', dpi=fig.dpi)

    def plot_avg_takeoff_top_5_airports(self, sc, df):
        df.createOrReplaceTempView("avg_takeoff_top_5_airports")
        temp_df = sc.sql("SELECT Year,ORIGIN,COUNT(DEP_TIME) as NumberOfFlightsFlown,AVG(TAXI_OUT) as AverageTakeOff from avg_takeoff_top_5_airports GROUP BY Year,ORIGIN")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql( \
            "WITH avg_takeoff_top_5_airports as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY NumberOfFlightsFlown desc) as rnk \
                FROM temp_df_sql ) \
            SELECT Year,ORIGIN, AverageTakeOff FROM avg_takeoff_top_5_airports WHERE rnk <= 5 GROUP BY Year,ORIGIN,AverageTakeOff ORDER BY Year,AverageTakeOff desc")
        pd_new_df = new_df.toPandas()
        print("Plotting Average TakeOff Time at Top 5 Airports for Each Year")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Average Take Off Time at Top 5 Airports for Each Year")
        sns.barplot(x=pd_new_df["Year"], y=pd_new_df["AverageTakeOff"], hue=pd_new_df["ORIGIN"], data=pd_new_df)
        plt.xlabel("Years")
        plt.ylabel("Average Take Off Time at Top 5 Airports in Minutes")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Airports, Average Take Off Time.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Airports, Average Take Off Time.png', dpi=fig.dpi)


    def plot_total_distancetravelled_top_5_carriers(self,sc,df):
        df.createOrReplaceTempView("total_travelldistance_top_5_carrier")
        temp_df = sc.sql(
            "SELECT OP_CARRIER,Year,COUNT(DEP_TIME) as NumberOfFlightsFlown,SUM(DISTANCE) as TotalDistanceTravelled from total_travelldistance_top_5_carrier GROUP BY OP_CARRIER,Year")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql( \
            "WITH total_travelldistance_top_5_carriers as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Year ORDER BY TotalDistanceTravelled desc) as rnk \
                FROM temp_df_sql ) \
            SELECT OP_CARRIER, Year, TotalDistanceTravelled FROM total_travelldistance_top_5_carriers WHERE rnk <= 5 ORDER BY Year")
        pd_new_df = new_df.toPandas()
        print("Plotting Total Distance Travelled for Top 5 Carriers for Each Year")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Total Distance Travelled for Top 5 Carriers for Each Year")
        sns.barplot(x=pd_new_df["OP_CARRIER"], y=pd_new_df["TotalDistanceTravelled"], hue=pd_new_df["Year"],
                    data=pd_new_df)
        plt.xlabel("Top 5 Carriers")
        plt.ylabel("Total Distance Travelled for Top 5 Carriers")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Carriers, Total Distance Flown.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Carriers, Total Distance Flown.png', dpi=fig.dpi)

    def plot_top5_desired_destination_states_byseason(self,sc,df):
        df.createOrReplaceTempView("desired_destinations_byseason")
        temp_df = sc.sql(
            "SELECT Seasons,DestinationState,COUNT(DEP_TIME) as NumberOfFlightsFlown\
             from desired_destinations_byseason GROUP BY Seasons,DestinationState")
        temp_df.createOrReplaceTempView("temp_df_sql")
        new_df = sc.sql( \
            "WITH desired_destinations_byseason_top_5 as ( \
                SELECT *, DENSE_RANK() OVER (PARTITION BY Seasons ORDER BY NumberOfFlightsFlown desc) as rnk \
                FROM temp_df_sql ) \
            SELECT Seasons, DestinationState, NumberOfFlightsFlown \
            FROM desired_destinations_byseason_top_5 WHERE rnk <= 5 ORDER BY Seasons")
        pd_new_df = new_df.toPandas()
        print("Plotting Top 5 Desired Destination States for Each Season")
        fig = plt.figure(figsize=(20, 10))
        plt.title("Top 5 Desired Destination States for Each Season")
        sns.barplot(x=pd_new_df["Seasons"], y=pd_new_df["NumberOfFlightsFlown"], hue=pd_new_df["DestinationState"],
                    data=pd_new_df)
        plt.xlabel("Seasons")
        plt.ylabel("Total Flights to Each Destination States")
        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Destination States By Season.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Destination States By Season.png', dpi=fig.dpi)


    def plot_piechart_parts_of_day(self,sc,df):
        df.createOrReplaceTempView("piechart_daypart")
        new_df = sc.sql("SELECT Parts_of_day, COUNT(*) as Number \
                         FROM piechart_daypart WHERE Parts_of_day is NOT NULL \
                         GROUP BY Parts_of_day  ORDER BY Number desc")
        fig = plt.figure(figsize=(20, 10))
        plt.pie(x=new_df.select(collect_list('Number')).first()[0], \
                labels= new_df.select(collect_list('Parts_of_day')).first()[0],autopct = '%1.2f%%')
        plt.title("Total Number of Flights at Different Parts of day using Pie Chart ")
       # plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Pie Chart for Parts of Day.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Pie Chart for Parts of Day.png', dpi=fig.dpi)


    def plot_top5_cities_asper_parts_of_day(self,sc,df):
        df.createOrReplaceTempView("top5_cities_parts_of_day")
        new_df = sc.sql("SELECT Parts_of_day, OriginCity, DestinationCity, COUNT(DEP_TIME) as OriginCount, \
                         COUNT(ARR_TIME) as DestCount  FROM top5_cities_parts_of_day GROUP BY Parts_of_day, \
                         OriginCity, DestinationCity ORDER BY Parts_of_day")

        new_df.createOrReplaceTempView("temp_df")
        temp_df_origin = sc.sql( \
                        "WITH origincity_top_5 as ( \
                            SELECT Parts_of_day, OriginCity, OriginCount,\
                             DENSE_RANK() OVER (PARTITION BY Parts_of_day ORDER BY OriginCount desc) as rnk \
                            FROM temp_df ) \
                        SELECT Parts_of_day, OriginCity, OriginCount \
                        FROM origincity_top_5 WHERE rnk <= 5 ORDER BY Parts_of_day")

        temp_df_dest = sc.sql( \
                        "WITH destcity_top_5 as ( \
                            SELECT Parts_of_day, DestinationCity, DestCount,\
                             DENSE_RANK() OVER (PARTITION BY Parts_of_day ORDER BY DestCount desc) as rnk \
                            FROM temp_df ) \
                        SELECT Parts_of_day, DestinationCity, DestCount \
                        FROM destcity_top_5 WHERE rnk <= 5 ORDER BY Parts_of_day")

        origin = temp_df_origin.toPandas()
        dest   = temp_df_dest.toPandas()

        plt.subplot(1, 2, 1)
        fig = plt.figure(figsize=(20, 10))
        plt.title("Top 5 Origin Cities for Each Part of Day")
        f1 = sns.barplot(x=origin["Parts_of_day"], y=origin["OriginCount"], hue=origin["OriginCity"],data=origin)
        plt.xlabel("Part of Day")
        plt.ylabel("Total Flights from Each Origin Cities")

        plt.subplot(1, 2, 2)
        fig = plt.figure(figsize=(20, 10))
        plt.title("Top 5 Destination Cities for Each Part of Day")
        f2 = sns.barplot(x=dest["Parts_of_day"], y=dest["DestCount"], hue=dest["DestinationCity"],data=dest)
        plt.xlabel("Part of Day")
        plt.ylabel("Total Flights to Each Destination Cities")

        #plt.show()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Top 5 Cities as Per Parts of Day.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #fig.savefig(S3_OUTPUT_PATH+'Top 5 Cities as Per Parts of Day.png', dpi=fig.dpi)


    def plot_boxplot(self,df):
        plt.subplot(1,2,1)

        x_arr = df.select(collect_list('ARR_DELAY')).first()[0]
        bp_arr = plt.boxplot(x_arr,showmeans=True)
        plt.title("Box Plot of Arrival Delay Time in Minutes")
        plt.xlabel("Time in Minutes")

        medians_arr  = [item.get_ydata()[0] for item in bp_arr['medians']]
        means_arr    = [item.get_ydata()[0] for item in bp_arr['means']]
        minimums_arr = [item.get_ydata()[0] for item in bp_arr['caps']][0]
        maximums_arr = [item.get_ydata()[0] for item in bp_arr['caps']][1]
        q1_arr       = [item.get_ydata()[0] for item in bp_arr['boxes']]
        q3_arr       = [item.get_ydata()[2] for item in bp_arr['boxes']]
        fliers_arr   = [item.get_ydata() for item in bp_arr['fliers']]
        lower_outliers_arr = []
        upper_outliers_arr = []
        for i in range(len(fliers_arr)):
            for outlier in fliers_arr[i]:
                if outlier < q1_arr[0]:
                    lower_outliers_arr.append(outlier)
                else:
                    upper_outliers_arr.append(outlier)
        print("For Arrival Delay Box Plot Information:")
        print("Minimum:",minimums_arr)
        print("Q1:",q1_arr)
        print("Median:",medians_arr)
        print("Mean:",means_arr)
        print("Q3:",q3_arr)
        print("Maximum:",maximums_arr)


        arr_bp = {"lower_outlier":lower_outliers_arr,"minimum":minimums_arr,"q1":q1_arr[0],\
                  "median":medians_arr[0],"mean":means_arr[0],"q3":q3_arr[0],"maximum":maximums_arr,\
                  "upper_outlier":upper_outliers_arr}


        plt.subplot(1,2,2)

        x_dep = df.select(collect_list('DEP_DELAY')).first()[0]
        bp_dep = plt.boxplot(x_dep,showmeans=True)
        plt.title("Box Plot of Departure Delay Time in Minutes")
        plt.xlabel("Time in Minutes")

        medians_dep  = [item.get_ydata()[0] for item in bp_dep['medians']]
        means_dep    = [item.get_ydata()[0] for item in bp_dep['means']]
        minimums_dep = [item.get_ydata()[0] for item in bp_dep['caps']][0]
        maximums_dep = [item.get_ydata()[0] for item in bp_dep['caps']][1]
        q1_dep       = [item.get_ydata()[0] for item in bp_dep['boxes']]
        q3_dep       = [item.get_ydata()[2] for item in bp_dep['boxes']]
        fliers_dep   = [item.get_ydata() for item in bp_dep['fliers']]
        lower_outliers_dep = []
        upper_outliers_dep = []
        for i in range(len(fliers_dep)):
            for outlier in fliers_dep[i]:
                if outlier < q1_dep[0]:
                    lower_outliers_dep.append(outlier)
                else:
                    upper_outliers_dep.append(outlier)
        print("\nFor Departure Delay Box Plot Information:")
        print("Minimum:",minimums_dep)
        print("Q1:",q1_dep)
        print("Median:",medians_dep)
        print("Mean:",means_dep)
        print("Q3:",q3_dep)
        print("Maximum:",maximums_dep)


        dep_bp = {"lower_outlier": lower_outliers_dep,"minimum":minimums_dep, "q1": q1_dep[0],\
                  "median": medians_dep[0], "mean": means_dep[0],"q3": q3_dep[0], "maximum":maximums_dep,\
                  "upper_outlier":upper_outliers_dep}


        #plt.show()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Box Plot.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #plt.savefig(S3_OUTPUT_PATH+'Box Plot.png')


        return arr_bp,dep_bp


    def plot_histogram_for_outliers(self,arr,dep):

        plt.subplot(1,2,1)
        plt.hist(dep, bins=10, edgecolor="black", color="green")
        plt.title("Histogram for Departure Delay Outliers")
        plt.xlabel("Time in Minutes")
        plt.ylabel("Frequency")


        plt.subplot(1,2,2)
        plt.hist(arr, bins=10, edgecolor="black", color="yellow")
        plt.title("Histogram for Arrival Delay Outliers")
        plt.xlabel("Time in Minutes")
        plt.ylabel("Frequency")


        #plt.show()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        s3 = s3fs.S3FileSystem(anon=False)  # Uses default credentials
        with s3.open(S3_OUTPUT_PATH + 'Histogram of Outliers.png', 'wb') as f:
            f.write(img_data.getbuffer())
        #plt.savefig(S3_OUTPUT_PATH+'Histogram of Outliers.png')


class DataPreparation:

    def create_labels(self,df):
        new_df = df.withColumn("Labels", when(df["ARR_DELAY"] < -10, lit("Early")).when(df["ARR_DELAY"] > 10,lit("Delay"))\
                               .otherwise(lit("OnTime")))

        return new_df

    def select_columns(self,df):
        new_df = df.select("OP_CARRIER","ORIGIN","DEST","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN",\
                           "DEP_TIME","DEP_DELAY","Labels")
        return new_df

    def split_data(self,df):
        train_df, test_df = df.randomSplit([0.8,0.2])
        return train_df, test_df

    def yearwise_df(self,sc,df,year):
        df.createOrReplaceTempView("temp_df")
        new_df = sc.sql("SELECT * FROM temp_df WHERE Year='{0}'".format(year))
        return new_df



class Model:

    def __init__(self,df):
        self.df = df
        cols = self.df.columns

        category_columns = [ "OP_CARRIER", "ORIGIN", "DEST"]
        indexoutputcols = [x + "_Index" for x in category_columns]
        oheoutputcols = [x + "_OneHot" for x in category_columns]
        numeric_col = ["TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","DEP_TIME","DEP_DELAY"]

        # Creating a String Indexer for Features Column
        self.stringIndexer = StringIndexer(inputCols=category_columns, outputCols=indexoutputcols, handleInvalid='skip') #---> On Local Pyspark version 3.1.3
        print("String Indexing Done")

        # Creating String Indexer for Labels Column
        self.labels_stringindexer = StringIndexer(inputCol='Labels', outputCol='label_index')
        print("Label String Indexing  Done")

        # Creating One Hot Encoder
        self.oheEncoder = OneHotEncoder(inputCols=indexoutputcols, outputCols=oheoutputcols) #----> On Local Pyspark 3.1.3
        print("One Hot Encoding Done")

        # Creating Vector Assembler
        assemblerinputs = oheoutputcols + numeric_col
        self.vectorassembler = VectorAssembler(inputCols=assemblerinputs, outputCol='features',handleInvalid='skip')
        print("Vector Assembling Done")

        self.stages = [self.stringIndexer, self.labels_stringindexer, self.oheEncoder, self.vectorassembler]
        print("Created a Stages Pipeline")

        stagespipeline = Pipeline(stages=self.stages)
        stagespipelinemodel = stagespipeline.fit(self.df)
        self.temp_df = stagespipelinemodel.transform(self.df)
        final_cols = cols + ['features','label_index']
        self.new_df = self.temp_df.select(final_cols)
        print("Stages PipeLine Transformed")

        self.train, self.test = self.new_df.randomSplit([0.7,0.3])
        print("Train Dataset Rows:",self.train.count())
        print("Test Dataset Rows:", self.test.count())

    def classification_accuracy(self,pred):
        pred.show(50,truncate=False)
        evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction")
        acc = evaluator.evaluate(pred)
        print("Accuracy = " , acc)

        return acc



    def decision_tree_classifier(self):
        start_time = time.time()
        #Creating Decision Tree Algorithm
        print("Using Decision Tree Classifier Algorithm")
        dt = DecisionTreeClassifier(labelCol='label_index',featuresCol='features',maxDepth=3)
        DT_Model = dt.fit(self.train)
        pred = DT_Model.transform(self.test)

        #Calculate Accuracy
        acc = self.classification_accuracy(pred)

        dt_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Decision Tree Algorithm Took: {dt_elapsed}")


        return acc,dt_elapsed



    def random_forest_classifier(self):
        start_time = time.time()
        #Creating Random Forest Algorithm
        print("Using Random Forest Algorithm")
        rf = RandomForestClassifier(labelCol='label_index',featuresCol='features')
        RF_Model = rf.fit(self.train)
        pred = RF_Model.transform(self.test)

        # Calculate Accuracy
        acc = self.classification_accuracy(pred)


        rf_lapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Random Forest Algorithm Took: {rf_elapsed}")

        return acc,rf_lapsed


    def multilayer_perceptron(self):
        start_time = time.time()
        print("Using Multi Layer Perceptron Algorithm")
        #Using Multi Layer Perceptron Algorithm

        # specify layers for the neural network:
        # input layer of size 797 (features), two intermediate of size 800 and 800
        # and output of size 3 (classes)
        layers = [732, 800, 800, 3]
        perceptron = MultilayerPerceptronClassifier(labelCol='label_index',featuresCol='features',maxIter=100, layers=layers, blockSize=128, seed=1234)
        MP_model = perceptron.fit(self.train)
        pred = MP_model.transform(self.test)

        # Calculate Accuracy
        acc = self.classification_accuracy(pred)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"MultiLayer Perceptron Took: {elapsed}")

        return acc


    def logistic_regression(self):
        start_time = time.time()
        print("Using Logistic Regression Algorithm")
        lr = LogisticRegression(featuresCol='features', labelCol='label_index', maxIter=10)
        ovr = OneVsRest(classifier=lr)
        ovrModel = ovr.fit(self.train)
        pred = ovrModel.transform(self.test)

        acc = self.classification_accuracy(pred)

        lr_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Logistic Reression Took: {lr_elapsed}")

        return acc,lr_elapsed

    def SVM(self):
        start_time = time.time()
        print("Using SVM Algorithm")
        lsvc = LinearSVC(featuresCol='features', labelCol='label_index',maxIter=10, regParam=0.1)
        ovr = OneVsRest(classifier=lsvc)
        ovrModel = ovr.fit(self.train)
        pred = ovrModel.transform(self.test)

        acc = self.classification_accuracy(pred)

        svm_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"SVM Took: {svm_elapsed}")

        return acc, svm_elapsed



if __name__ == "__main__":
    #Creating a SparkSession Object
    create_session_obj = CreateSession()
    spark_session = create_session_obj.create_session()

    #Creating a Object for Reading data
    read_data_obj = ImportData()

######################################################################################
    # print("2009 CSV File")
    # initial_df_2009 = read_data_obj.read_csv(spark_session, 'data\\2009.csv')
    # print("2010 CSV File")
    # initial_df_2010 = read_data_obj.read_csv(spark_session, 'data\\2010.csv')
    # print("2011 CSV File")
    # initial_df_2011 = read_data_obj.read_csv(spark_session,'data\\2011.csv')
    # print("2012 CSV File")
    # initial_df_2012 = read_data_obj.read_csv(spark_session, 'data\\2012.csv')
    # print("2013 CSV File")
    # initial_df_2013 = read_data_obj.read_csv(spark_session, 'data\\2013.csv')
    # print("2014 CSV File")
    # initial_df_2014 = read_data_obj.read_csv(spark_session, 'data\\2014.csv')
    # print("2015 CSV File")
    # initial_df_2015 = read_data_obj.read_csv(spark_session, 'data\\2015.csv')
    # print("2016 CSV File")
    # initial_df_2016 = read_data_obj.read_csv(spark_session, 'data\\2016.csv')
    # print("2017 CSV File")
    # initial_df_2017 = read_data_obj.read_csv(spark_session, 'data\\2017.csv')
    # print("2018 CSV File")
    # initial_df_2018 = read_data_obj.read_csv(spark_session, 'data\\2018.csv')
    # initial_df_airport_state = read_data_obj.read_csv(spark_session, 'data\\temp.csv')

    print("2009 CSV File")
    initial_df_2009 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2009.csv')
    print("2010 CSV File")
    initial_df_2010 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2010.csv')
    print("2011 CSV File")
    initial_df_2011 = read_data_obj.read_csv(spark_session,S3_BUCKET_READ_DATA_PATH +'\\2011.csv')
    print("2012 CSV File")
    initial_df_2012 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2012.csv')
    print("2013 CSV File")
    initial_df_2013 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2013.csv')
    print("2014 CSV File")
    initial_df_2014 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2014.csv')
    print("2015 CSV File")
    initial_df_2015 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2015.csv')
    print("2016 CSV File")
    initial_df_2016 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2016.csv')
    print("2017 CSV File")
    initial_df_2017 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2017.csv')
    print("2018 CSV File")
    initial_df_2018 = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\2018.csv')

    initial_df_airport_state = read_data_obj.read_csv(spark_session, S3_BUCKET_READ_DATA_PATH +'\\temp.csv')
######################################################################################3


    #Creating Feature Engineering Object
    feature_engi_obj = FeatureEngineering()


######################################################################################3
    #Count Null Values
    print("Percentage of Null Values in 2009 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2009)
    print("Percentage of Null Values in 2010 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2010)
    print("Percentage of Null Values in 2011 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2011)
    print("Percentage of Null Values in 2012 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2012)
    print("Percentage of Null Values in 2013 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2013)
    print("Percentage of Null Values in 2014 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2014)
    print("Percentage of Null Values in 2015 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2015)
    print("Percentage of Null Values in 2016 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2016)
    print("Percentage of Null Values in 2017 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2017)
    print("Percentage of Null Values in 2018 Year for Each Column:")
    feature_engi_obj.count_null(initial_df_2018)
######################################################################################3
    """
    Since Last 6 columns has more than 80% of Null values we will drop those columns has they serve no Importance
    """
    #Dropping Columns which serves no importance
    new_2009 = feature_engi_obj.drop_columns(initial_df_2009, columns=initial_df_2009.columns[-6:])
    new_2010 = feature_engi_obj.drop_columns(initial_df_2010, columns=initial_df_2010.columns[-6:])
    new_2011 = feature_engi_obj.drop_columns(initial_df_2011, columns=initial_df_2011.columns[-6:])
    new_2012 = feature_engi_obj.drop_columns(initial_df_2012, columns=initial_df_2012.columns[-6:])
    new_2013 = feature_engi_obj.drop_columns(initial_df_2013, columns=initial_df_2013.columns[-6:])
    new_2014 = feature_engi_obj.drop_columns(initial_df_2014, columns=initial_df_2014.columns[-6:])
    new_2015 = feature_engi_obj.drop_columns(initial_df_2015, columns=initial_df_2015.columns[-6:])
    new_2016 = feature_engi_obj.drop_columns(initial_df_2016, columns=initial_df_2016.columns[-6:])
    new_2017 = feature_engi_obj.drop_columns(initial_df_2017, columns=initial_df_2017.columns[-6:])
    new_2018 = feature_engi_obj.drop_columns(initial_df_2018, columns=initial_df_2018.columns[-6:])

##########################################################################

    #Mapping State and City
    print("Mapping State and City for 2009 year")
    new_2009_map = feature_engi_obj.map_state_city(new_2009,initial_df_airport_state)
    print("Mapping State and City for 2010 year")
    new_2010_map = feature_engi_obj.map_state_city(new_2010,initial_df_airport_state)
    print("Mapping State and City for 2011 year")
    new_2011_map = feature_engi_obj.map_state_city(new_2011,initial_df_airport_state)
    print("Mapping State and City for 2012 year")
    new_2012_map = feature_engi_obj.map_state_city(new_2012,initial_df_airport_state)
    print("Mapping State and City for 2013 year")
    new_2013_map = feature_engi_obj.map_state_city(new_2013,initial_df_airport_state)
    print("Mapping State and City for 2014 year")
    new_2014_map = feature_engi_obj.map_state_city(new_2014,initial_df_airport_state)
    print("Mapping State and City for 2015 year")
    new_2015_map = feature_engi_obj.map_state_city(new_2015,initial_df_airport_state)
    print("Mapping State and City for 2016 year")
    new_2016_map = feature_engi_obj.map_state_city(new_2016,initial_df_airport_state)
    print("Mapping State and City for 2017 year")
    new_2017_map = feature_engi_obj.map_state_city(new_2017,initial_df_airport_state)
    print("Mapping State and City for 2018 year")
    new_2018_map = feature_engi_obj.map_state_city(new_2018,initial_df_airport_state)

####################################################################################
    #Extarcting Year and Month
    print("Extracting Year and Month for 2009 year")
    new_2009_year_month = feature_engi_obj.extract_year_month(new_2009_map)
    print("Extracting Year and Month for 2010 year")
    new_2010_year_month = feature_engi_obj.extract_year_month(new_2010_map)
    print("Extracting Year and Month for 2011 year")
    new_2011_year_month = feature_engi_obj.extract_year_month(new_2011_map)
    print("Extracting Year and Month for 2012 year")
    new_2012_year_month = feature_engi_obj.extract_year_month(new_2012_map)
    print("Extracting Year and Month for 2013 year")
    new_2013_year_month = feature_engi_obj.extract_year_month(new_2013_map)
    print("Extracting Year and Month for 2014 year")
    new_2014_year_month = feature_engi_obj.extract_year_month(new_2014_map)
    print("Extracting Year and Month for 2015 year")
    new_2015_year_month = feature_engi_obj.extract_year_month(new_2015_map)
    print("Extracting Year and Month for 2016 year")
    new_2016_year_month = feature_engi_obj.extract_year_month(new_2016_map)
    print("Extracting Year and Month for 2017 year")
    new_2017_year_month = feature_engi_obj.extract_year_month(new_2017_map)
    print("Extracting Year and Month for 2018 year")
    new_2018_year_month = feature_engi_obj.extract_year_month(new_2018_map)
#############################################################################
    # Extarcting Seasons
    print("Extarcting Seasons for 2009 year")
    new_2009_seasons = feature_engi_obj.seasons(spark_session,new_2009_year_month)
    print("Extarcting Seasons for 2010 year")
    new_2010_seasons = feature_engi_obj.seasons(spark_session,new_2010_year_month)
    print("Extarcting Seasons for 2011 year")
    new_2011_seasons = feature_engi_obj.seasons(spark_session,new_2011_year_month)
    print("Extarcting Seasons for 2012 year")
    new_2012_seasons = feature_engi_obj.seasons(spark_session,new_2012_year_month)
    print("Extarcting Seasons for 2013 year")
    new_2013_seasons = feature_engi_obj.seasons(spark_session,new_2013_year_month)
    print("Extarcting Seasons for 2014 year")
    new_2014_seasons = feature_engi_obj.seasons(spark_session,new_2014_year_month)
    print("Extarcting Seasons for 2015 year")
    new_2015_seasons = feature_engi_obj.seasons(spark_session,new_2015_year_month)
    print("Extarcting Seasons for 2016 year")
    new_2016_seasons = feature_engi_obj.seasons(spark_session,new_2016_year_month)
    print("Extarcting Seasons for 2017 year")
    new_2017_seasons = feature_engi_obj.seasons(spark_session,new_2017_year_month)
    print("Extarcting Seasons for 2018 year")
    new_2018_seasons = feature_engi_obj.seasons(spark_session,new_2018_year_month)
#############################################################################
    # Extarcting Parts of Day
    print("Extarcting Parts of Day for 2009 year")
    new_2009_daypart = feature_engi_obj.parts_of_day(spark_session,new_2009_seasons)
    print("Extarcting Parts of Day for 2010 year")
    new_2010_daypart = feature_engi_obj.parts_of_day(spark_session,new_2010_seasons)
    print("Extarcting Parts of Day for 2011 year")
    new_2011_daypart = feature_engi_obj.parts_of_day(spark_session,new_2011_seasons)
    print("Extarcting Parts of Day for 2012 year")
    new_2012_daypart = feature_engi_obj.parts_of_day(spark_session,new_2012_seasons)
    print("Extarcting Parts of Day for 2013 year")
    new_2013_daypart = feature_engi_obj.parts_of_day(spark_session,new_2013_seasons)
    print("Extarcting Parts of Day for 2014 year")
    new_2014_daypart = feature_engi_obj.parts_of_day(spark_session,new_2014_seasons)
    print("Extarcting Parts of Day for 2015 year")
    new_2015_daypart = feature_engi_obj.parts_of_day(spark_session,new_2015_seasons)
    print("Extarcting Parts of Day for 2016 year")
    new_2016_daypart = feature_engi_obj.parts_of_day(spark_session,new_2016_seasons)
    print("Extarcting Parts of Day for 2017 year")
    new_2017_daypart = feature_engi_obj.parts_of_day(spark_session,new_2017_seasons)
    print("Extarcting Parts of Day for 2018 year")
    new_2018_daypart = feature_engi_obj.parts_of_day(spark_session,new_2018_seasons)
########################################################################################
    #Combined DataFrame for Future Use
    combined_df = feature_engi_obj.combine_all_dataframes(new_2009_daypart,new_2010_daypart,new_2011_daypart,\
                                                          new_2012_daypart,new_2013_daypart,new_2014_daypart,\
                                                          new_2015_daypart,new_2016_daypart,new_2017_daypart,\
                                                          new_2018_daypart)

    combined_df = feature_engi_obj.combine_all_dataframes(new_2017_daypart,new_2018_daypart)
# # ########################################################################################
    """
    Data Visualizations
    """
    data_visualize_obj = DataVisualization()
    # 2009
    data_visualize_obj.plot_no_flights_per_month(spark_session,new_2009_daypart,'2009')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session,new_2009_daypart,'2009')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session,new_2009_daypart,'2009')
    data_visualize_obj.plot_most_desired_day(spark_session,new_2009_daypart,'2009')
    
    
    
    # 2010
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2010_daypart,'2010')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2010_daypart,'2010')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2010_daypart,'2010')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2010_daypart,'2010')
    
    
    
    # 2011
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2011_daypart,'2011')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2011_daypart,'2011')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2011_daypart,'2011')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2011_daypart,'2011')
    
    
    
    # 2012
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2012_daypart,'2012')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2012_daypart,'2012')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2012_daypart,'2012')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2012_daypart,'2012')
    
    
    
    # 2013
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2013_daypart,'2013')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2013_daypart,'2013')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2013_daypart,'2013')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2013_daypart,'2013')
    
    
    
    # 2014
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2014_daypart,'2014')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2014_daypart,'2014')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2014_daypart,'2014')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2014_daypart,'2014')
    
    
    
    # 2015
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2015_daypart,'2015')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2015_daypart,'2015')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2015_daypart,'2015')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2015_daypart,'2015')
    
    
    
    # 2016
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2016_daypart,'2016')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2016_daypart,'2016')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2016_daypart,'2016')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2016_daypart,'2016')
    
    
    
    # 2017
    data_visualize_obj.plot_no_flights_per_month(spark_session, new_2017_daypart,'2017')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session, new_2017_daypart,'2017')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session, new_2017_daypart,'2017')
    data_visualize_obj.plot_most_desired_day(spark_session, new_2017_daypart,'2017')
    
    
    
    #2018
    data_visualize_obj.plot_no_flights_per_month(spark_session,new_2018_daypart,'2018')
    data_visualize_obj.plot_no_flights_by_each_carrier(spark_session,new_2018_daypart,'2018')
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session,new_2018_daypart,'2018')
    data_visualize_obj.plot_most_desired_day(spark_session,new_2018_daypart,'2018')

    # Top 5 Carriers for Each Year
    data_visualize_obj.plot_top_5_carriers(spark_session,combined_df)

    # Top 5 Destiantion States for Each Year
    data_visualize_obj.plot_top_5_most_desired_destination_states_each_year(spark_session,combined_df)

    # AVG Departure Delay And Arrival Delay Time of Top 5 Carriers
    data_visualize_obj.plot_avg_departuredelay_arrivaldelay_top_5_carrier(spark_session,combined_df)
    
    #AVG Departure Delay Time of Top 5 Airports
    data_visualize_obj.plot_avg_departure_delaytime_top_5_airports(spark_session,combined_df)
    
    #AVG Take Off Time of Top 5 Airports
    data_visualize_obj.plot_avg_takeoff_top_5_airports(spark_session,combined_df)

    # Total Distance Travelled by Top 5 Carrier
    data_visualize_obj.plot_total_distancetravelled_top_5_carriers(spark_session,combined_df)

    # Top 10 Desired Destiantion States for Each Year
    data_visualize_obj.plot_top_10_most_desired_destination_states(spark_session,combined_df)

    # Top 5 Desired Destiantion City for Custom State for Each Year
    data_visualize_obj.plot_top_5_most_desired_destination_city_customstate_each_year(spark_session,combined_df,'California')
    
    # Plot Total Flights for 2 States
    data_visualize_obj.plot_most_desired_destination_customstate1_customestate2_each_year(spark_session,combined_df,'California','Texas')

    # Plot Total Flights for Top 5 Carriers for Each Day
    data_visualize_obj.plot_total_flights_top_5_carriers_eachday(spark_session,combined_df)
    
    # Plot Total Flights for Top 5 Carriers for Each Day
    data_visualize_obj.plot_top5_desired_destination_states_byseason(spark_session,combined_df)
    
    #Total Number of Flights at Different Parts of Day Using Pie Chart
    data_visualize_obj.plot_piechart_parts_of_day(spark_session,combined_df)
    
    # Top 5 Cities at Each Time of the Day
    data_visualize_obj.plot_top5_cities_asper_parts_of_day(spark_session,combined_df)

    # Plotting Top 15 Most Desired Departure Location
    data_visualize_obj.plot_most_desired_departure_location(spark_session,combined_df)

############################################################################################
    #Feature Engineering for Model
    combined_df_without_cancellationcode_1 = feature_engi_obj.cancellationcode_without_1(combined_df)
    # combined_df_without_cancellationcode_1 = feature_engi_obj.cancellationcode_without_1(new_2018_daypart)



    # Plot Box Plot
    # arrival_bp,departure_bp = data_visualize_obj.plot_boxplot(initial_df_2013)
    arrival_bp, departure_bp = data_visualize_obj.plot_boxplot(combined_df_without_cancellationcode_1)

    # Plot Histogram
    data_visualize_obj.plot_histogram_for_outliers(arrival_bp["upper_outlier"],departure_bp["upper_outlier"])

#   Since any value after 400 minutes was less hence we will drop those rows \
#   with values greater than 400 for both arr-delay and dep_delay

    combined_df_removed_outliers = feature_engi_obj.remove_outliers(combined_df_without_cancellationcode_1)

########################################################################################################
    # Data Preparation Stage
    data_preparation_obj = DataPreparation()

    # Create Labels
    combined_df_labels = data_preparation_obj.create_labels(combined_df_removed_outliers)
    print("Labels Created")

    # Select Columns for Model
    combined_df_columns = data_preparation_obj.select_columns(combined_df_labels)

    #Splitting the Data According to Year
    df_2009 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2009')
    df_2010 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2010')
    df_2011 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2011')
    df_2012 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2012')
    df_2013 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2013')
    df_2014 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2014')
    df_2015 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2015')
    df_2016 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2016')
    df_2017 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2017')
    df_2018 = data_preparation_obj.yearwise_df(spark_session, combined_df_columns, '2018')
    print("Data Splitted according to Year")

    #80-20 Split According to Data
    combined_train_df_2009, combined_test_df_2009 = data_preparation_obj.split_data(df_2009)
    combined_train_df_2010, combined_test_df_2010 = data_preparation_obj.split_data(df_2010)
    combined_train_df_2011, combined_test_df_2011 = data_preparation_obj.split_data(df_2011)
    combined_train_df_2012, combined_test_df_2012 = data_preparation_obj.split_data(df_2012)
    combined_train_df_2013, combined_test_df_2013 = data_preparation_obj.split_data(df_2013)
    combined_train_df_2014, combined_test_df_2014 = data_preparation_obj.split_data(df_2014)
    combined_train_df_2015, combined_test_df_2015 = data_preparation_obj.split_data(df_2015)
    combined_train_df_2016, combined_test_df_2016 = data_preparation_obj.split_data(df_2016)
    combined_train_df_2017, combined_test_df_2017 = data_preparation_obj.split_data(df_2017)
    combined_train_df_2018, combined_test_df_2018 = data_preparation_obj.split_data(df_2018)

    # Split Data
    combined_train_df, combined_test_df = data_preparation_obj.split_data(combined_df_columns)
    print("Data Splitted")

    combined_train_df = feature_engi_obj.combine_all_dataframes(combined_train_df_2009,combined_train_df_2010,combined_train_df_2011,\
                                                          combined_train_df_2012,combined_train_df_2013,combined_train_df_2014,\
                                                          combined_train_df_2015,combined_train_df_2016,combined_train_df_2017,\
                                                          combined_train_df_2018)
    
    combined_test_df = feature_engi_obj.combine_all_dataframes(combined_test_df_2009,combined_test_df_2010,combined_test_df_2011,\
                                                          combined_test_df_2012,combined_test_df_2013,combined_test_df_2014,\
                                                          combined_test_df_2015,combined_test_df_2016,combined_test_df_2017,\
                                                          combined_test_df_2018)


    print("Train DataSet:",combined_train_df.count())
    print("Test DataSet:",combined_test_df.count())


#########################################################################################################

    #Model Preparation
    model = Model(combined_df_columns)

    # Decision Tree Classifier
    dt_acc,dt_time = model.decision_tree_classifier()

    # Random Forest Classifier
    rf_acc,rf_time = model.random_forest_classifier()

    # Multi Layer Perceptron
    #mp_acc = model.multilayer_perceptron()

    #Logistic Regression
    lr_acc,lr_time = model.logistic_regression()

    #SVM
    svm_acc,svm_time = model.SVM()

    result_matrix = pd.DataFrame(data={"Algorithm":["Decision Tree Algorithm","Random Forest Algorithm",\
                                       "Logistic Regression Algorithm","SVM Classifier"],
                                       "Accuracy":[dt_acc,rf_acc,lr_acc,svm_acc],\
                                       "Time Spent":[dt_time,rf_time,lr_time,svm_time]})

    print(result_matrix)







#####################################################################################
    main_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - main_start_time))
    print(f"Entire Code Took: {main_elapsed}")
##########################################################################################

    input("Press Enter to end SparkSession:")
    spark_session.stop()
