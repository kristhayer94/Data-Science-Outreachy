# Databricks notebook source
# MAGIC %md
# MAGIC # Heavy Users Over Time - No Search Counts

# COMMAND ----------

# MAGIC %md
# MAGIC ####Heavy User Definition

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis of Heavy Users can be seen in this notebook: <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/70791/command/70792">Heavy User Cutoffs</a>

# COMMAND ----------

# MAGIC %md
# MAGIC Look at Heavy Users over a week and Heavy User Retention.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as st
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ###Prepare Data

# COMMAND ----------

#Global variables
sample_id = 42
hu_start_date = '20180926'  # Calculations will look back 6 days before this for week average
number_days = 7
num_weeks = 6

# COMMAND ----------

sum_query = """
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri, 
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)*5/3600) AS td_active_hours,
      first(sample_id) AS sample_id
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    GROUP BY
        1, 2
    """

search_query = """
  SELECT client_id,
         submission_date_s3,
         engine,
         SUM(sap) as sap,
         SUM(tagged_sap) as tagged_sap,
         SUM(tagged_follow_on) as tagged_follow_on,
         SUM(organic) as in_content_organic
  FROM search_clients_daily
  WHERE
      submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
  GROUP BY
      1, 2, 3
    """

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get multiple weeks of cutoffs

# COMMAND ----------

# Get a week start and end date for number_days consecutive days
wk_st_end = []
num_wk_days = num_weeks * 7
for x in range(0, num_wk_days+1):
    wk_st_end.append([(pd.to_datetime(hu_start_date) + pd.DateOffset(days=x-6)).strftime('%Y%m%d'), 
                      (pd.to_datetime(hu_start_date) + pd.DateOffset(days=x)).strftime('%Y%m%d')])

# COMMAND ----------

wk_st_end

# COMMAND ----------

# This takes a long time
# for the date ranges defined above
ms_wk = []
perc80 = [.8]
error = 0
uri_cutoff = []
sc_cutoff = []
ah_cutoff = []
for i in range(0, len(wk_st_end)):
    # Get the week's data for an endday
    ms_sum_wk = spark.sql(sum_query.format(wk_st_end[i][0],wk_st_end[i][1],sample_id))
    search_wk = spark.sql(search_query.format(wk_st_end[i][0],wk_st_end[i][1],sample_id))    
    search_wk = search_wk.na.fill(0)
    search_sum_wk = search_wk.groupBy('client_id', 'submission_date_s3') \
                         .agg(F.sum(F.col('sap') + F.col('in_content_organic'))) \
                         .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts') \
                         .sort('client_id', 'submission_date_s3')
    # Join the main summary and search data for the week
    ms_week = ms_sum_wk.join(search_sum_wk, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)

    # Add the week's data to the list of weeks
    ms_wk.append(ms_week)
    # Get the aDAU values for the week
    ms_wk_aDAU = ms_week.where('td_uri >= 5')
    # Average the aDAU values for the week
    ms_wk_avg_aDAU = ms_wk_aDAU.groupBy('client_id').avg() \
        .withColumnRenamed('avg(td_uri)','avg_uri') \
        .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
        .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
        .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')
    
    # Get the 80th percentile cutoff for the heavy user type for day and add it to the list of cutoffs
    uri_cutoff.append(ms_wk_avg_aDAU.stat.approxQuantile('avg_uri', perc80, error)[0])
    sc_cutoff.append(ms_wk_avg_aDAU.stat.approxQuantile('avg_search_counts', perc80, error)[0])
    ah_cutoff.append(ms_wk_avg_aDAU.stat.approxQuantile('avg_active_hours', perc80, error)[0])


# COMMAND ----------

uri_cutoff

# COMMAND ----------

sc_cutoff

# COMMAND ----------

ah_cutoff

# COMMAND ----------

# Define day labels for graphs
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daylbl = []
for i in range(0, number_days): 
  wk_end_dt = datetime.strptime(wk_st_end[i][1], '%Y%m%d')
  daylbl.append(days[wk_end_dt.weekday()]+'\n'+wk_end_dt.strftime('%m/%d/%Y'))

# COMMAND ----------

# Graph the URI cutoffs over the consecutive days defined in wk_st_end
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(number_days), uri_cutoff[0:number_days])
ax.set_ylabel('URI')
ax.set_ylim(165, 172)
xticks = array(list(range(0, number_days)))
ax.set_xticks(xticks)
ax.set_xticklabels(daylbl)
ax.set_title(r'URI Cutoffs Over a Week')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# Graph the Search Count cutoffs over the consecutive days defined in wk_st_end
#plt.gcf().clear()
#fig, ax = plt.subplots(figsize=(10,6))
#ax.plot(range(number_days), sc_cutoff[0:number_days])
#ax.set_ylabel('Search Counts')
#ax.set_title(r'Search Count Cutoffs Over a Week')
#xticks = array(list(range(0, number_days)))
#ax.set_xticks(xticks)
#ax.set_xticklabels(daylbl)
#ax.legend(loc='best')
#
#display(fig)

# COMMAND ----------

# Graph the Active Hour cutoffs over the consecutive days defined in wk_st_end
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(number_days), ah_cutoff[0:number_days])
ax.set_ylabel('Active Hours')
ax.set_ylim(0.85, 1)
xticks = array(list(range(0, number_days)))
ax.set_xticks(xticks)
ax.set_xticklabels(daylbl)
ax.set_title(r'Active Hour Cutoffs Over a Week')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of Heavy Users for a week

# COMMAND ----------

# This takes longer than average cells
# Get arrays of number of heavy user clients for a day
ms_day = []
day_hu_uri = []
day_hu_sc = []
day_hu_ah = []
uri_num_clients_day = []
sc_num_clients_day = []
ah_num_clients_day = []
# For the list of consecutive days
for i in range(0, number_days):
    print "i " + str(i) + " date: " + wk_st_end[i][1]
    # Get the data from the last day of the week and add it to the list of day data
    ms_day.append(ms_wk[i].where("submission_date_s3 = \'" +  wk_st_end[i][1] + "\'"))

    # Get the data for the heavy users from the day and add it to the list of day heavy users
    day_hu_uri.append(ms_day[i].where('td_uri >= ' + str(uri_cutoff[i])))
    day_hu_sc.append(ms_day[i].where('td_search_counts > '  + str(sc_cutoff[i])))  
    day_hu_ah.append(ms_day[i].where('td_active_hours > '  + str(ah_cutoff[i])))
    
    # Get the number of clients for the heavy users from the day and add it to the list of number of clients
    uri_num_clients_day.append(day_hu_uri[i].select('client_id').distinct().count())
    sc_num_clients_day.append(day_hu_sc[i].select('client_id').distinct().count())
    ah_num_clients_day.append(day_hu_ah[i].select('client_id').distinct().count())

    # Combine the day's heavy user data with the previous days into one big dataframe
    if (i == 1):
        hu_uri = day_hu_uri[i-1].union(day_hu_uri[i])
        hu_sc = day_hu_sc[i-1].union(day_hu_sc[i])
        hu_ah = day_hu_ah[i-1].union(day_hu_ah[i])
    elif (i > 1):
        hu_uri = hu_uri.union(day_hu_uri[i])
        hu_sc = hu_sc.union(day_hu_sc[i])
        hu_ah = hu_ah.union(day_hu_ah[i])

# COMMAND ----------

uri_num_clients_day

# COMMAND ----------

# Generate bar chart for the number of heavy users for a day of the week
width = 0.25
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the bar chart of the data
xticks = array(list(range(0, number_days)))
ax.bar(xticks - width, uri_num_clients_day[0:number_days], width, label = 'Heavy URI Clients', color = 'deepskyblue')
ax.bar(xticks, sc_num_clients_day[0:number_days], width, label = 'Heavy Search Count Clients', color = 'lightsalmon')
ax.bar(xticks + width, ah_num_clients_day[0:number_days], width, label = 'Heavy Active Hour Clients', color = 'mediumaquamarine')

ax.set_xticks(xticks)
ax.set_xticklabels(daylbl)
ax.set_ylim(0, 260000)
#ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Clients')
ax.set_title(r'Number of Heavy Users per Day')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

display(hu_uri.select(['submission_date_s3', 'td_uri', 'td_active_hours']).describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of days a week for heavy use

# COMMAND ----------

# Count all the days for heavy use by uri
huuri_day_count = hu_uri.groupby(['client_id']).count().withColumnRenamed('count','num_days')
# Count the number of clients for a number of days
huuri_freq = huuri_day_count.groupby('num_days') \
      .agg(F.countDistinct('client_id')).withColumnRenamed('count(DISTINCT client_id)', 'uri_num_clients').sort('num_days')

# Count all the days for heavy use by search count
husc_day_count = hu_sc.groupby(['client_id']).count().withColumnRenamed('count','num_days')
# Count the number of clients for a number of days
husc_freq = husc_day_count.groupby('num_days') \
      .agg(F.countDistinct('client_id')).withColumnRenamed('count(DISTINCT client_id)', 'sc_num_clients').sort('num_days')

# Count all the days for heavy use by active hours
huah_day_count = hu_ah.groupby(['client_id']).count().withColumnRenamed('count','num_days')
# Count the number of clients for a number of days
huah_freq = huah_day_count.groupby('num_days') \
      .agg(F.countDistinct('client_id')).withColumnRenamed('count(DISTINCT client_id)', 'ah_num_clients').sort('num_days')

# COMMAND ----------

display(huuri_day_count)

# COMMAND ----------

display(huuri_freq)

# COMMAND ----------

# Generate bar chart
num_day = array(huuri_freq.select('num_days').rdd.flatMap(lambda x:x).collect())
uri_num_clients_arr = array(huuri_freq.select('uri_num_clients').rdd.flatMap(lambda x: x).collect())
sc_num_clients_arr = array(husc_freq.select('sc_num_clients').rdd.flatMap(lambda x: x).collect())
ah_num_clients_arr = array(huah_freq.select('ah_num_clients').rdd.flatMap(lambda x: x).collect())

width = 0.25
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the bar chart of the data
rects_uri = ax.bar(num_day - width, uri_num_clients_arr, width, label = 'Heavy URI Clients', color = 'deepskyblue')
rects_sc = ax.bar(num_day, sc_num_clients_arr, width, label = 'Heavy Search Count Clients', color = 'lightsalmon')
rects_ah = ax.bar(num_day + width, ah_num_clients_arr, width, label = 'Heavy Active Hour Clients', color = 'mediumaquamarine')

ax.set_xlabel('Number of Days of the Week')
ax.set_ylabel('Number of Clients')
ax.set_title(r'Number of Days of the Week for Heavy Users')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC If there are similar amounts of heavy users each day of the week, how come the heavy user counts fall off so much for more than 1 day a week?
# MAGIC 
# MAGIC There are different heavy users each day of the week.

# COMMAND ----------

# Wednesday's URI heavy users
display(day_hu_uri[0].select(['submission_date_s3', 'td_uri', 'td_active_hours']).describe())

# COMMAND ----------

# Thursday's URI heavy users
display(day_hu_uri[1].select(['submission_date_s3', 'td_uri', 'td_active_hours']).describe())

# COMMAND ----------

# URI Heavy Users on Wednesday but not Thursday
only_w = day_hu_uri[0].join(day_hu_uri[1], 'client_id', 'left_anti')
display(only_w.select(['submission_date_s3', 'td_uri', 'td_active_hours']).describe())

# COMMAND ----------

# URI Heavy Users on Thursday but not Wednesday
only_th = day_hu_uri[1].join(day_hu_uri[0], 'client_id', 'left_anti')
display(only_th.select(['submission_date_s3', 'td_uri', 'td_active_hours']).describe())

# COMMAND ----------

# URI Heavy Users on Wednesday who are also on Thursday
both = day_hu_uri[0].join(day_hu_uri[1], 'client_id')

# COMMAND ----------

only_w_cnt = only_w.count()
only_th_cnt = only_th.count()
both_cnt = both.count()
print 'Heavy Users on Wednesday but not Thursday ' + str(only_w_cnt)
print 'Heavy Users on Thursday but not Wednesday ' + str(only_th_cnt)
print 'Heavy Users both days ' + str(both_cnt)

# COMMAND ----------

# Get all the URI heavy users in the week with 1 day of heavy use
huuri_1day = huuri_day_count.where('num_days == 1')
uri_1day = hu_uri.join(huuri_1day, 'client_id')
display(uri_1day.select(['submission_date_s3', 'td_uri', 'td_active_hours', 'num_days']))

# COMMAND ----------

husc_1day = husc_day_count.where('num_days == 1')
sc_1day = hu_sc.join(husc_1day, 'client_id')

huah_1day = huah_day_count.where('num_days == 1')
ah_1day = hu_ah.join(huah_1day, 'client_id')

# Count the number of heavy users each day for clients with only 1 day of heavy use
uri_1day_count = uri_1day.groupby(['submission_date_s3']).count().withColumnRenamed('count','num_clients').sort('submission_date_s3')
sc_1day_count = sc_1day.groupby(['submission_date_s3']).count().withColumnRenamed('count','num_clients').sort('submission_date_s3')
ah_1day_count = ah_1day.groupby(['submission_date_s3']).count().withColumnRenamed('count','num_clients').sort('submission_date_s3')

# COMMAND ----------

uri_1day_clients_arr = array(uri_1day_count.select('num_clients').rdd.flatMap(lambda x: x).collect())
sc_1day_clients_arr = array(sc_1day_count.select('num_clients').rdd.flatMap(lambda x: x).collect())
ah_1day_clients_arr = array(ah_1day_count.select('num_clients').rdd.flatMap(lambda x: x).collect())

# COMMAND ----------

# Generate bar chart
width = 0.25
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the bar chart of the data
xticks = array(list(range(0, number_days)))
#ax.bar(xticks, uri_1day_clients_arr, width, label = 'Heavy URI Clients', color = 'deepskyblue')
ax.bar(xticks - width, uri_1day_clients_arr[0:number_days], width, label = 'Heavy URI Clients', color = 'deepskyblue')
ax.bar(xticks, sc_1day_clients_arr[0:number_days], width, label = 'Heavy Search Count Clients', color = 'lightsalmon')
ax.bar(xticks + width, ah_1day_clients_arr[0:number_days], width, label = 'Heavy Active Hour Clients', color = 'mediumaquamarine')

ax.set_xticks(xticks)
ax.set_xticklabels(daylbl)
#ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Clients')
ax.set_title(r'Number of 1 Day Heavy Users')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retention

# COMMAND ----------

PERIODS = {}
N_WEEKS = 6
for i in range(1, N_WEEKS + 1):
    PERIODS[i] = {
        'start': i * 7,
        'end': i * 7 + 6
    }  

# COMMAND ----------

import datetime as dt
import pandas as pd
import pyspark.sql.types as st
import pyspark.sql.functions as F

udf = F.udf

def date_diff(d1, d2, fmt='%Y%m%d'):
    """
    Returns days elapsed from d2 to d1 as an integer

    Params:
    d1 (str)
    d2 (str)
    fmt (str): format of d1 and d2 (must be the same)

    >>> date_diff('20170205', '20170201')
    4

    >>> date_diff('20170201', '20170205)
    -4
    """
    try:
        return (pd.to_datetime(d1, format=fmt) -
                pd.to_datetime(d2, format=fmt)).days
    except:
        return None


@udf(returnType=st.IntegerType())
def get_period(anchor, submission_date_s3):
    """
    Given an anchor and a submission_date_s3,
    returns what period a ping belongs to. This
    is a spark UDF.

    Params:
    anchor (col): anchor date
    submission_date_s3 (col): a ping's submission_date to s3

    Global:
    PERIODS (dict): defined globally based on n-week method

    Returns an integer indicating the retention period
    """
    if anchor is not None:
        diff = date_diff(submission_date_s3, anchor)
        if diff >= 7: # exclude first 7 days
            for period in sorted(PERIODS):
                if diff <= PERIODS[period]['end']:
                    return period


# COMMAND ----------

# Start of the baseline week
base_start = wk_st_end[0][0]
base_end = wk_st_end[0][1]
# Last date for the number of weeks
num_week_days = num_weeks * 7
weekx_end = wk_st_end[num_week_days-1][1]

# COMMAND ----------

co_dates = []
for i in range(0, len(wk_st_end)):
    co_dates.append(wk_st_end[i][1])
co_dates

# COMMAND ----------

ms_xweek = spark.sql("""
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri, 
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)*5/3600) AS td_active_hours,
      first(sample_id) AS sample_id
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    GROUP BY
        1, 2
    """.format(base_start, weekx_end, sample_id)
    )

search_xweek = spark.sql("""
  SELECT client_id,
       submission_date_s3,
       engine,
       SUM(sap) as sap,
       SUM(tagged_sap) as tagged_sap,
       SUM(tagged_follow_on) as tagged_follow_on,
       SUM(organic) as in_content_organic
  FROM search_clients_daily
  WHERE
      submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
  GROUP BY
      1, 2, 3
    """.format(base_start, weekx_end, sample_id)
    )

# COMMAND ----------

search_xweek = search_xweek.na.fill(0)
sc_xweek = search_xweek.groupBy('client_id', 'submission_date_s3') \
                     .agg(F.sum(F.col('sap') + F.col('in_content_organic'))) \
                     .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts') \
                     .sort('client_id', 'submission_date_s3')
# Join the main summary and search data for the week
ms_ret = ms_xweek.join(sc_xweek, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)

# COMMAND ----------

display(ms_ret.select(['submission_date_s3', 'td_uri', 'td_active_hours', 'sample_id']).describe())

# COMMAND ----------

def calc_retention(heavy_col_name, heavy_co_arr, heavy_co_date, anchor_start, anchor_end):
    """
    Given a dataframe, heavy column name, heavy cutoff, and base week end date,
    returns dataframe with retention for heavy user type for each period.
    
    Params:
    heavy_col_name (string): name of the column for heavy user
    heavy_co_arr (array numbers): an array of cutoff values to define a heavy user for a date
    heavy_co_date (array dates): an array of dates for the cutoff values
    anchor_start (date): date for the start of the anchor week
    anchor_end (date): date for the end of the anchor week
    
    Returns dataframe with retention for heavy user type for each period
    """
  
    # Get heavy users
    for i in range(0, len(heavy_co_date)):
        if i == 0:
            hu_where = "(submission_date_s3 <= " + heavy_co_date[i] + " and " + heavy_col_name + " >= " + str(heavy_co_arr[i]) + ")"
        else:
            hu_where = hu_where + " or (submission_date_s3 = " + heavy_co_date[i] + " and " + heavy_col_name + " >= " + str(heavy_co_arr[i]) + ")"
    print hu_where
            
    ms_ret_hu = ms_ret.where(hu_where)
    # Get base week of heavy users
    base_hu = ms_ret_hu \
        .where("submission_date_s3 >= \'" +  anchor_start + "\' and submission_date_s3 <= \'" +  anchor_end + "\'") \
        .select('client_id').distinct()
    # Count unique clients in base week of heavy users
    base_num_clients = base_hu.select('client_id').distinct().count()
    # Get only base week heavy users for the entire time
    ret_hu = ms_ret_hu.join(base_hu, on='client_id')
    # Add period column
    ret_hu = ret_hu.withColumn("anchor", F.lit(base_start)) \
                   .withColumn("period", get_period("anchor", "submission_date_s3"))
    # Group by period and count the number of distinct clients
    hu_weekly_counts = ret_hu.groupby("period").agg(F.countDistinct("client_id").alias("n_week_clients"))
    # Add column with calculated retention and 95% CI
    hu_retention = (
        hu_weekly_counts
            .withColumn("total_clients", F.lit(base_num_clients))
            .withColumn("retention", F.col("n_week_clients") / F.col("total_clients"))
            # Add a 95% confidence interval based on the normal approximation for a binomial distribution,
            # p ± z * sqrt(p*(1-p)/n).
            # The 95% CI spans the range `retention ± ci_95_semi_interval`.
            .withColumn(
                "ci_95_semi_interval",
                F.lit(1.96) * F.sqrt(F.col("retention") * (F.lit(1) - F.col("retention")) / F.col("total_clients"))
            )
    )
    return hu_retention

# COMMAND ----------

huuri_retention = calc_retention('td_uri', uri_cutoff, co_dates, base_start, base_end)
huuri_ret = huuri_retention['period', 'retention', 'ci_95_semi_interval'].where("retention < 1").sort('period')
display(huuri_ret)

# COMMAND ----------

huah_retention = calc_retention('td_active_hours', ah_cutoff, co_dates, base_start, base_end)
huah_ret = huah_retention['period', 'retention', 'ci_95_semi_interval'].where("retention < 1").sort('period')
display(huah_ret)

# COMMAND ----------

husc_retention = calc_retention('td_search_counts', sc_cutoff, co_dates, base_start, base_end)
husc_ret = husc_retention['period', 'retention', 'ci_95_semi_interval'].where("retention < 1").sort('period')
display(husc_ret)

# COMMAND ----------

weeks = array(huuri_ret.select('period').rdd.flatMap(lambda x:x).collect())
uri_ret = array(huuri_ret.select('retention').rdd.flatMap(lambda x: x).collect())
sc_ret = array(husc_ret.select('retention').rdd.flatMap(lambda x: x).collect())
ah_ret = array(huah_ret.select('retention').rdd.flatMap(lambda x: x).collect())


# COMMAND ----------

# Graph retention over multiple weeks
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(weeks, uri_ret, label='URI')
ax.plot(weeks, sc_ret, label='Search Counts')
ax.plot(weeks, ah_ret, label='Active Hours')
ax.set_ylabel('Percentage')
ax.set_ylim(0, 1)
ax.set_xlabel('Weeks')
ax.set_title('Retention over ' + str(num_weeks) + ' Weeks')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attributes of Heavy Users

# COMMAND ----------

# MAGIC %md
# MAGIC The attributes of Heavy Users can be seen in this notebook: <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/73994/command/76992">Heavy User Attributes</a>

# COMMAND ----------


