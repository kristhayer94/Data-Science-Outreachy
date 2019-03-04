# Databricks notebook source
# MAGIC %md
# MAGIC # Heavy User Cutoffs - No Search Counts

# COMMAND ----------

# MAGIC %md
# MAGIC Based on analysis done by Brendan Colloran from the Strategy and Insights team in 2016, Saptarshi Guha in 2017 and Project Ahab in 2018, I will be looking at URI count, search count, subsession hours and active hours.
# MAGIC 
# MAGIC I'm looking at data from a week in September as a baseline since that avoids summer and major holidays.
# MAGIC 
# MAGIC I'm reading the Firefox data from main_summary.  I could use clients_daily instead of adding up a day's worth of pings from main_summary.  The advantage to this would be that I wouldn't have to create the work of clients_daily.  The downside is that I've spent most of my time analyzing main_summary data so I'm more familiar with it, and I've noticed some high values for active_hours in clients_daily that I can't account for from the pings in main_summary.  My analysis of the difference between main_summary and clients_daily is in this notebook: <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/72034/command/72063">Main Summary vs Clients Daily</a>

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as st
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ###Prepare Data

# COMMAND ----------

#Global variables
sample_id = 42
week_1_start = '20180920'
week_1_end = '20180926'
week_4_end = '20181017'
thday = '20180920'
fday = '20180921'
saday = '20180922'
sday = '20180923'
mday = '20180924'
tday = '20180925'
wday = '20180926'
sday_cond = "submission_date_s3 = \'" +  sday + "\'"
mday_cond = "submission_date_s3 = \'" +  mday + "\'"
tday_cond = "submission_date_s3 = \'" +  tday + "\'"
wday_cond = "submission_date_s3 = \'" +  wday + "\'"
thday_cond = "submission_date_s3 = \'" +  thday + "\'"
fday_cond = "submission_date_s3 = \'" +  fday + "\'"
saday_cond = "submission_date_s3 = \'" +  saday + "\'"
prob = (0.5, 0.75, 0.80, 0.90, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99, 0.995)
relError = 0
percentiles = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.0]
bp_day_lbls = ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday']
bp_seg_lbls = ['Day All', 'Day No Zeros', 'Day No Zeros No Ex', 'Day aDAU', 'Week Avg All', 'Week Avg No Zeros', 
             'Week Avg No Zeros No Ex', 'Week Avg aDAU', '4 Week Avg aDAU']

# COMMAND ----------

# MAGIC %md
# MAGIC Active ticks are in 5 second increments, so it is converted to hours. Subsession length is in seconds, so it is converted to hours. In looking at the difference between active_ticks and scalar_parent_browser_engagement_active_ticks, they are the same values except where active_ticks has a number scalar_parent_browser_engagement_active_ticks is often null.
# MAGIC 
# MAGIC Calculate the values for the day from pings the same way the clients_daily value determines them - sum, mean, max, first, count.
# MAGIC 
# MAGIC For search counts, per Ben's suggestion, using search_clients_daily. "For total searches, you can use sap for "searches issued from Mozilla's UI", and organic for "searches made in content" (i.e. from google.com's ui)"

# COMMAND ----------

ping_query = """
    SELECT
        client_id,
        submission_date_s3,
        coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri_count,
        coalesce(scalar_parent_browser_engagement_active_ticks, 0) AS active_ticks,
        (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_hours,        
        subsession_length,
        (subsession_length/3600) AS subsession_hours,
        session_length,
        profile_subsession_counter,
        subsession_counter,
        session_start_date,
        subsession_start_date,
        reason,
        active_addons_count,
        scalar_parent_browser_engagement_max_concurrent_tab_count AS tab_count,
        scalar_parent_browser_engagement_max_concurrent_window_count AS window_count,
        scalar_parent_browser_engagement_unique_domains_count AS domains_count,
        profile_creation_date,
        profile_reset_date,
        previous_build_id,
        normalized_channel,
        os,
        normalized_os_version,
        windows_build_number,
        install_year,
        creation_date,
        distribution_id,
        submission_date,
        app_build_id,
        app_display_version,
        update_channel,
        update_enabled,
        update_auto_download,
        timezone_offset,
        vendor,
        is_default_browser,
        default_search_engine,
        devtools_toolbox_opened_count,
        client_submission_date,
        places_bookmarks_count,
        places_pages_count,
        scalar_parent_browser_engagement_tab_open_event_count AS tab_event_count,
        scalar_parent_browser_engagement_window_open_event_count AS window_event_count,
        scalar_parent_browser_errors_collected_count AS errors_collected_count,
        scalar_parent_devtools_current_theme AS current_theme,
        scalar_parent_formautofill_availability AS formautofill_availability, 
        country,
        city,
        geo_subdivision1,
        locale,
        antivirus,
        antispyware,
        firewall,
        session_id,
        subsession_id,
        sync_configured,
        sync_count_desktop,
        sync_count_mobile,
        disabled_addons_ids,
        active_theme,
        user_prefs,
        experiments,
        sample_id,
        document_id
    FROM main_summary
    WHERE 
      app_name='Firefox'
      AND submission_date_s3 >= '{}'
      AND submission_date_s3 <= '{}'
      AND sample_id = '{}'
    ORDER BY
        client_id,
        submission_date_s3,
        profile_subsession_counter
    """

# From telemetry docs for how clients_daily deteremines values
sum_query = """
    SELECT 
      client_id,
      submission_date_s3,
      sum(coalesce(scalar_parent_browser_engagement_total_uri_count, 0)) AS td_uri,
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)) AS td_active_ticks,      
      sum(coalesce(scalar_parent_browser_engagement_active_ticks, 0)*5/3600) AS td_active_hours,
      sum(subsession_length/3600) AS td_subsession_hours,   
      sum(CASE WHEN subsession_counter = 1 THEN 1 ELSE 0 END) AS sessions_started_on_this_day,
      mean(active_addons_count) AS active_addons_count_mean,
      max(scalar_parent_browser_engagement_max_concurrent_tab_count) AS tab_count_max,
      max(scalar_parent_browser_engagement_max_concurrent_window_count) AS window_count_max,
      max(scalar_parent_browser_engagement_unique_domains_count) AS domains_count_max,
      first(profile_creation_date) AS profile_creation_date,
      first(previous_build_id) AS previous_build_id,
      first(normalized_channel) AS normalized_channel,
      first(os) AS os,
      first(normalized_os_version) AS normalized_os_version,
      first(windows_build_number) AS windows_build_number,
      first(install_year) AS install_year,
      first(distribution_id) AS distribution_id,
      count(distinct document_id) AS pings_aggregated_by_this_row,
      first(app_build_id) AS app_build_id,
      first(app_display_version) AS app_display_version,
      first(update_channel) AS update_channel,
      first(update_enabled) AS update_enabled,
      first(update_auto_download) AS update_auto_download,
      first(timezone_offset) AS timezone_offset,
      first(vendor) AS vendor,
      first(is_default_browser) AS is_default_browser,
      first(default_search_engine) AS default_search_engine,
      sum(devtools_toolbox_opened_count) AS devtools_toolbox_opened_count_sum,
      mean(places_bookmarks_count) AS places_bookmarks_count_mean,
      mean(places_pages_count) AS places_pages_count_mean,
      sum(scalar_parent_browser_engagement_tab_open_event_count) AS td_tab_event_count,
      sum(scalar_parent_browser_engagement_window_open_event_count) AS td_window_event_count,
      first(CASE WHEN country IS NOT NULL AND country != '??' THEN country ELSE NULL END) as country,
      first(CASE WHEN country IS NOT NULL AND country != '??' 
                THEN CASE WHEN city IS NOT NULL THEN city ELSE '??' END
                ELSE NULL END) AS city,
      first(geo_subdivision1) AS geo_subdivision1,
      first(locale) AS locale,
      first(sync_configured) AS sync_configured,
      sum(sync_count_desktop) AS sync_count_desktop,
      sum(sync_count_mobile) AS sync_count_mobile,
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

# Execute queries
ms_1week_ping = spark.sql(ping_query.format(week_1_start, week_1_end, sample_id))
ms_1week_sum = spark.sql(sum_query.format(week_1_start,week_1_end,sample_id))
search_wk = spark.sql(search_query.format(week_1_start,week_1_end,sample_id))

ms_4week_sum = spark.sql(sum_query.format(week_1_start,week_4_end,sample_id))
search_4wk = spark.sql(search_query.format(week_1_start,week_4_end,sample_id))

# COMMAND ----------

# Fill null search counts with 0s
search_wk = search_wk.na.fill(0)
# Sum over the day
search_wk_sum = search_wk.groupBy('client_id', 'submission_date_s3') \
                         .agg(F.sum(F.col('sap')+F.col('in_content_organic')))  \
                         .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts') \
                         .sort('client_id', 'submission_date_s3')
# Join the search table with the other counts
ms_1week = ms_1week_sum.join(search_wk_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)

# Average the total daily values over the week
# This averages over the days that have data, not over 7 days
ms_1week_avg = ms_1week.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('avg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')

# COMMAND ----------

# Look at all clients summed over a day with search counts
ms_1week_cols = ms_1week.columns
ms_1week_cols.remove('client_id')
ms_1week_cols.remove('td_search_counts')
display(ms_1week.sort('client_id', 'submission_date_s3').select(ms_1week_cols))

# COMMAND ----------

# Count all client day records for a week and all distinct clients for a week
week_count = ms_1week.count()
num_all_clients = ms_1week.select('client_id').distinct().count()

# COMMAND ----------

# Look at all client records averaged over a week (for the days with records, not over 7 days)
ms_1week_avg_cols = ms_1week_avg.columns
ms_1week_avg_cols.remove('client_id')
ms_1week_avg_cols.remove('avg_search_counts')
display(ms_1week_avg.select(ms_1week_avg_cols))

# COMMAND ----------

# Get data for 28 days, then average over days with records
# Fill null search counts with 0s
search_4wk = search_4wk.na.fill(0)
# Sum over the day
search_4wk_sum = search_4wk.groupBy('client_id', 'submission_date_s3') \
                           .agg(F.sum(F.col('sap')+F.col('in_content_organic')))  \
                           .withColumnRenamed('sum((sap + in_content_organic))','td_search_counts') \
                           .sort('client_id', 'submission_date_s3')
# Join the search table with the other counts
ms_4week = ms_4week_sum.join(search_4wk_sum, ['client_id', 'submission_date_s3'], 'full_outer').na.fill(0)

# Average the total daily values over the week
# This averages over the days that have data, not over 7 days
ms_4week_avg = ms_4week.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('avg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Decide if we should use active_hours or subsession_hours
# MAGIC 
# MAGIC From Brendan Colloran's analysis from 2016: Session hours are less than ideal for a number of reasons -- they include idle time (FF running in the background) and can even include time that the computer is asleep, and there is a known anomalies in the measurement that can cause more than 24 session hours to be counted for a single calendar date. But the URI count probes were not ready as of Jan 2017, and the other potentially best option would be something like active ticks, which has a nebulously bad reputation for being “unreliable”.
# MAGIC 
# MAGIC But ultimately this may not matter that much -- we know that session hours and pageviews are correlated, and there is no reason to believe that active ticks is not correlated with these metrics as well, and we’re just choosing an arbitrary threshold on some value with the intent of paying more attention to the “heavy” end of our user base rather than the overall “average” or “modal” user. The top 10% of any of these measures should be sufficient for that, even if any of these probes has some wrinkles.

# COMMAND ----------

# MAGIC %md
# MAGIC Based on my analysis of active hours compared to subsesion hours in this notebook,  <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/73398/command/73429">Active Hours vs Subsesion Hours</a>, I have chosen to use active hours as the time measurement for heavy users.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine quantile values for URIs, search counts, and active hours
# MAGIC 
# MAGIC Look at range and outliers for an individual day and averaged over a week (for the days with data, not over 7 days) for:  
# MAGIC * all records
# MAGIC * records where uri and active hours are both > 0
# MAGIC * records where uri and active hours are both > 0 and uri and search counts are both < extremes
# MAGIC * only aDAU users

# COMMAND ----------

# MAGIC %md
# MAGIC #### All records

# COMMAND ----------

# Get 1 day of summed values from the week
ms_thday = ms_1week.where(thday_cond)
ms_fday = ms_1week.where(fday_cond)
ms_saday = ms_1week.where(saday_cond)
ms_sday = ms_1week.where(sday_cond)
ms_mday = ms_1week.where(mday_cond)
ms_tday = ms_1week.where(tday_cond)
ms_wday = ms_1week.where(wday_cond)
num_wday_clients = ms_wday.count()
display(ms_wday.select(ms_1week_cols))

# COMMAND ----------

display(ms_1week['submission_date_s3', 'td_uri', 'td_active_hours'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Records where uri and active hours are both > 0  

# COMMAND ----------

# Get client days where uri count = 0 and active hours = 0
ms_1week_zero = ms_1week.where('td_uri == 0 and td_active_hours == 0')
display(ms_1week_zero.select(ms_1week_cols).describe())

# COMMAND ----------

# Calculate percentage of client day records that have zeros for both uri and active_hours
week_zero_count = ms_1week_zero.count()
print '%0.2f'%((week_zero_count *1.0/week_count) * 100), '% of the client day records have zeros for both uri and active hours'

# COMMAND ----------

# Number of clients with zero URI and active hour records
zero_clients = ms_1week_zero.select('client_id').distinct().count()
zero_clients

# COMMAND ----------

# Get client day records where uri and active hours are greater than 0
ms_1week_nz = ms_1week.where('td_uri > 0 and td_active_hours > 0')
display(ms_1week_nz.select(ms_1week_cols).describe())

# COMMAND ----------

# Average the non zero daily values over the week
# This averages over the days that have data, not over 7 days
ms_1week_avg_nz = ms_1week_nz.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('avg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')
#display(ms_1week_avg_nz)

# COMMAND ----------

# Get 1 day of summed values from the week
ms_wday_nz = ms_1week_nz.where(wday_cond)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Records where URI and active hours are both > 0  and below extreme outliers
# MAGIC URI < 1500

# COMMAND ----------

# Extreme URI counts
ms_1week_exuri = ms_1week_nz.where('td_uri >= 1500')
display(ms_1week_exuri.select(ms_1week_cols).describe())

# COMMAND ----------

# Get client day records where uri >= 1500
ms_1week_extreme = ms_1week_nz.where('td_uri >= 1500')
display(ms_1week_extreme.select(ms_1week_cols).describe())

# COMMAND ----------

# Calculate percentage of client day records that have extreme outliers
week_extreme_count = ms_1week_extreme.count()
print '%0.2f'%((week_extreme_count *1.0/week_count) * 100), '% of the client day records have extreme outliers'

# COMMAND ----------

# Number of clients with extreme records
extreme_clients = ms_1week_extreme.select('client_id').distinct().count()
extreme_clients

# COMMAND ----------

# Get client day records with uri and active ticks > 0 and uri and search counts extremes
ms_1week_nznx = ms_1week.where('td_uri > 0 and td_active_hours > 0 and td_uri < 1500')
display(ms_1week_nznx.select(ms_1week_cols).describe())

# COMMAND ----------

# Average the non zero daily non extreme values over the week
# This averages over the days that have data, not over 7 days
ms_1week_avg_nznx = ms_1week_nznx.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('anxvg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')
#display(ms_1week_avg_nznx)

# COMMAND ----------

# Get 1 day of summed values from the week
ms_wday_nznx = ms_1week_nznx.where(wday_cond)

# COMMAND ----------

# MAGIC %md
# MAGIC #### aDAU Records

# COMMAND ----------

# Get client day records where uri >= 5
ms_1week_aDAU = ms_1week.where('td_uri >= 5')
display(ms_1week_aDAU.select(ms_1week_cols).describe())

# COMMAND ----------

# Calculate percentage of client day records that have uri >= 5
week_aDAU_count = ms_1week_aDAU.count()
print '%0.2f'%((week_aDAU_count *1.0/week_count) * 100), '% of the client day records have uri >= 5'

# COMMAND ----------

# Number of aDAU clients
aDAU_clients = ms_1week_aDAU.select('client_id').distinct().count()
aDAU_clients

# COMMAND ----------

# Average the aDAU values over the week
# This averages over the days that have data, not over 7 days
ms_1week_avg_aDAU = ms_1week_aDAU.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('anxvg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')
#display(ms_1week_avg_aDAU)

# COMMAND ----------

# Get 1 day of summed values from the week
ms_wday_aDAU = ms_1week_aDAU.where(wday_cond)

# COMMAND ----------

# Get client day records where uri >= 5
ms_4week_aDAU = ms_4week.where('td_uri >= 5')
# Average the daily values over 4 weeks
# This averages over the days that have data, not over 7 days
ms_4week_avg_aDAU = ms_4week_aDAU.groupBy('client_id').avg() \
    .withColumnRenamed('avg(td_uri)','avg_uri') \
    .withColumnRenamed('avg(td_active_ticks)','avg_active_ticks') \
    .withColumnRenamed('avg(td_active_hours)','avg_active_hours') \
    .withColumnRenamed('avg(td_subsession_hours)','avg_subsession_hours') \
    .withColumnRenamed('avg(sessions_started_on_this_day)','avg_sessions_started') \
    .withColumnRenamed('avg(active_addons_count_mean)','avg_addons_count') \
    .withColumnRenamed('avg(tab_count_max)','avg_tab_count') \
    .withColumnRenamed('avg(window_count_max)','avg_window_count') \
    .withColumnRenamed('avg(domains_count_max)','avg_domains_count') \
    .withColumnRenamed('avg(pings_aggregated_by_this_row)', 'avg_pings') \
    .withColumnRenamed('anxvg(sync_count_desktop)', 'avg_sync_desktop') \
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile') \
    .withColumnRenamed('avg(td_search_counts)', 'avg_search_counts')

# COMMAND ----------

def get_day_arrays(column_name):
    thday_all = np.array(ms_thday.select(column_name).rdd.flatMap(lambda x: x).collect())
    fday_all = np.array(ms_fday.select(column_name).rdd.flatMap(lambda x: x).collect())
    saday_all = np.array(ms_saday.select(column_name).rdd.flatMap(lambda x: x).collect())
    sday_all = np.array(ms_sday.select(column_name).rdd.flatMap(lambda x: x).collect())
    mday_all = np.array(ms_mday.select(column_name).rdd.flatMap(lambda x: x).collect())
    tday_all = np.array(ms_tday.select(column_name).rdd.flatMap(lambda x: x).collect())
    wday_all = np.array(ms_wday.select(column_name).rdd.flatMap(lambda x: x).collect())

    day_arrs = [thday_all, fday_all, saday_all, sday_all, mday_all, tday_all, wday_all]
    return day_arrs
  
def get_seg_arrays(day_col_name, avg_col_name):
    wday_all = np.array(ms_wday.select(day_col_name).rdd.flatMap(lambda x: x).collect())
    wday_nz = np.array(ms_wday_nz.select(day_col_name).rdd.flatMap(lambda x: x).collect())
    wday_nznx = np.array(ms_wday_nznx.select(day_col_name).rdd.flatMap(lambda x: x).collect())
    wday_aDAU = np.array(ms_wday_aDAU.select(day_col_name).rdd.flatMap(lambda x: x).collect())    
    avg_week_all = np.array(ms_1week_avg.select(avg_col_name).rdd.flatMap(lambda x: x).collect())
    avg_week_nz = np.array(ms_1week_avg_nz.select(avg_col_name).rdd.flatMap(lambda x: x).collect())
    avg_week_nznx = np.array(ms_1week_avg_nznx.select(avg_col_name).rdd.flatMap(lambda x: x).collect())
    avg_week_aDAU = np.array(ms_1week_avg_aDAU.select(avg_col_name).rdd.flatMap(lambda x: x).collect())
    avg_4week_aDAU = np.array(ms_4week_avg_aDAU.select(avg_col_name).rdd.flatMap(lambda x: x).collect())
    
    seg_arrs = [wday_all, wday_nz, wday_nznx, wday_aDAU, avg_week_all, avg_week_nz, avg_week_nznx, avg_week_aDAU, avg_4week_aDAU]
    return seg_arrs
  
def get_quant_arrays(day_col_name, avg_col_name):
    day_all_quant = ms_wday.stat.approxQuantile(day_col_name, percentiles, relError)
    day_nz_quant = ms_wday_nz.stat.approxQuantile(day_col_name, percentiles, relError)
    day_nznx_quant = ms_wday_nznx.stat.approxQuantile(day_col_name, percentiles, relError)
    day_aDAU_quant = ms_wday_aDAU.stat.approxQuantile(day_col_name, percentiles, relError)
    week_all_quant = ms_1week_avg.stat.approxQuantile(avg_col_name, percentiles, relError)
    week_nz_quant = ms_1week_avg_nz.stat.approxQuantile(avg_col_name, percentiles, relError)
    week_nznx_quant = ms_1week_avg_nznx.stat.approxQuantile(avg_col_name, percentiles, relError)
    week_aDAU_quant = ms_1week_avg_aDAU.stat.approxQuantile(avg_col_name, percentiles, relError)
    week4_aDAU_quant = ms_4week_avg_aDAU.stat.approxQuantile(avg_col_name, percentiles, relError)

    quant_arrs = [day_all_quant, day_nz_quant, day_nznx_quant, day_aDAU_quant, week_all_quant, 
                  week_nz_quant, week_nznx_quant, week_aDAU_quant, week4_aDAU_quant]
    return quant_arrs

cols = ['Percentile', 'Day All', 'Day Nz', 'Day NzNx', 'Day aDAU', 'Week All', 'Week Nz', 'Week NzNx', 'Week aDAU', '4 Weeks aDAU']  
def get_quant_df(day_col_name, avg_col_name):  
    percentiles = pd.DataFrame({'Percentile': prob,
                   'Day All': ms_wday.stat.approxQuantile(day_col_name, prob, relError),
                   'Day Nz': ms_wday_nz.stat.approxQuantile(day_col_name, prob, relError),
                   'Day NzNx': ms_wday_nznx.stat.approxQuantile(day_col_name, prob, relError),
                   'Day aDAU': ms_wday_aDAU.stat.approxQuantile(day_col_name, prob, relError),
                   'Week All': ms_1week_avg.stat.approxQuantile(avg_col_name, prob, relError),
                   'Week Nz': ms_1week_avg_nz.stat.approxQuantile(avg_col_name, prob, relError),
                   'Week NzNx': ms_1week_avg_nznx.stat.approxQuantile(avg_col_name, prob, relError),
                   'Week aDAU': ms_1week_avg_aDAU.stat.approxQuantile(avg_col_name, prob, relError),                                
                   '4 Weeks aDAU': ms_4week_avg_aDAU.stat.approxQuantile(avg_col_name, prob, relError),                   
                  })
    percentiles['Percentile'] = percentiles['Percentile']*100
    # for all the columns starting at Day All, apply format for 2 decimal places
    for col in cols[cols.index('Day All'):] :
        percentiles[col] = percentiles[col].apply('{:.2f}'.format)
    # order the columns
    percentiles = percentiles[cols]
    return percentiles

# COMMAND ----------

def draw_box_plots(distarr, data, title, ylabel, yllim, yulim):
    # Draw box plots for 7 days
    plt.gcf().clear()
    fig, ax1 = plt.subplots(figsize=(10,6))
    #bp = ax1.boxplot(data, sym='', whis=[5, 95])
    bp = ax1.boxplot(data, sym='')
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(yllim, yulim)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)  
    ax1.set_xticklabels(distarr, rotation=10, fontsize=8)
    means = [np.mean(x) for x in data]
    plt.scatter(range(1, len(distarr)+1), means)
    display(fig)  
  
    return None

def draw_quantiles(data, title, ylabel, yllim, yulim):  
    plt.gcf().clear()
    fig, ax1 = plt.subplots(figsize=(10,6))
    xticks = range(0, len(percentiles), 1)
    ax1.plot(data[0], label='All - Day')
    ax1.plot(data[1], '--', label='No Zero - Day')
    ax1.plot(data[2], '--',label='No Zero No Ex - Day')
    ax1.plot(data[3], '--', label='aDAU - Day')
    ax1.plot(data[4], label='All - Week')
    ax1.plot(data[5], ':', label='No Zero - Week')
    ax1.plot(data[6], ':', label='No Zero No Ex - Week')
    ax1.plot(data[7], ':', label='aDAU - Week')
    ax1.plot(data[8], label='aDAU - 4 Weeks')
    ax1.set_title(title)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(percentiles)
    ax1.set_ylim(yllim, yulim)
    ax1.set_xlabel('Quantile')
    ax1.set_ylabel(ylabel)
    ax1.legend(loc='best')
    display(fig)
    return None
                 

# COMMAND ----------

# MAGIC %md
# MAGIC #### URI Ranges and Quantiles

# COMMAND ----------

day_data = get_day_arrays('td_uri')
draw_box_plots(bp_day_lbls, day_data, 'URI Counts for Each Day of a Week', 'URI', -5, 350)

# COMMAND ----------

seg_data = get_seg_arrays('td_uri', 'avg_uri')
draw_box_plots(bp_seg_lbls, seg_data, 'URI Counts for Different Segments of Data', 'URI', -5, 410)

# COMMAND ----------

quant_data = get_quant_arrays('td_uri', 'avg_uri')
draw_quantiles(quant_data, 'URI Quantiles', 'URI', -5, 500)

# COMMAND ----------

uri_percentiles = get_quant_df('td_uri', 'avg_uri')
display(uri_percentiles)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Search Counts Ranges and Quantiles

# COMMAND ----------

day_data = get_day_arrays('td_search_counts')
draw_box_plots(bp_day_lbls, day_data, 'Search Counts for Each Day of a Week', 'Search Counts', -.25, 1200)

# COMMAND ----------

seg_data = get_seg_arrays('td_search_counts', 'avg_search_counts')
draw_box_plots(bp_seg_lbls, seg_data, 'Search Counts for Different Segments of Data', 'Search Counts', -.25, 1200)

# COMMAND ----------

quant_data = get_quant_arrays('td_search_counts', 'avg_search_counts')
draw_quantiles(quant_data, 'Search Count Quantiles', 'Search Counts', -1, 1200)

# COMMAND ----------

sc_percentiles = get_quant_df('td_search_counts', 'avg_search_counts')
display(sc_percentiles)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Active Hours Ranges and Quantiles

# COMMAND ----------

day_data = get_day_arrays('td_active_hours')
draw_box_plots(bp_day_lbls, day_data, 'Active Hours for Each Day of a Week', 'Active Hours', -.1, 2.5)

# COMMAND ----------

seg_data = get_seg_arrays('td_active_hours', 'avg_active_hours')
draw_box_plots(bp_seg_lbls, seg_data, 'Active Hours for Different Segments of Data', 'Active Hours', -.1, 2.5)

# COMMAND ----------

quant_data = get_quant_arrays('td_active_hours', 'avg_active_hours')
draw_quantiles(quant_data, 'Active Hour Quantiles', 'Active Hours', 0, 3)

# COMMAND ----------

ah_percentiles = get_quant_df('td_active_hours', 'avg_active_hours')
display(ah_percentiles)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlations

# COMMAND ----------

def plot_scatter_line(ax, xarray, xlabel, xlim, yarray, ylabel, ylim):
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xlim)
    ax.scatter(xarray, yarray, s=1)
    ax.plot(np.unique(xarray), np.poly1d(np.polyfit(xarray, yarray, 1))(np.unique(xarray)), color='y')
    
    return None

# COMMAND ----------

# three scatter plots: uri - sc, uri - ah, sc - ah
active_hrs = np.array(ms_wday.rdd.map(lambda p: p.td_active_hours).collect())
uri = np.array(ms_wday.rdd.map(lambda p: p.td_uri).collect())
search_counts = np.array(ms_wday.rdd.map(lambda p: p.td_search_counts).collect())

plt.gcf().clear()
fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))

#plot_scatter_line(ax1, search_counts, 'Search Counts', 1200, uri, 'URI', 2000)
plot_scatter_line(ax2, active_hrs, 'Active Hours', 20, uri, 'URI', 2000)
#plot_scatter_line(ax3, active_hrs, 'Active Hours', 20, search_counts, 'Search Counts', 1200)

plt.tight_layout()
display(fig)


# COMMAND ----------

print 'Correlation of search counts to uri is', ms_wday.stat.corr('td_search_counts', 'td_uri')
print 'Correlation of active hours to uri is', ms_wday.stat.corr('td_active_hours', 'td_uri')
print 'Correlation of active hours to search counts is', ms_wday.stat.corr('td_active_hours', 'td_search_counts')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proposed heavy user  
# MAGIC 
# MAGIC The “heavy-half” concept  generally suggests that 80% of the volume of a product is consumed by 20% of its consumers (Twedt 1964)  
# MAGIC https://en.ryte.com/wiki/Heavy_User
# MAGIC 
# MAGIC In Project Ahab they considered users based on time, intensity (uri) and revenue (search).
# MAGIC 
# MAGIC 3 Heavy User types
# MAGIC * 80th percentile or higher in active hours 
# MAGIC * 80th percentile or higher in uri 
# MAGIC * 80th percentile or higher in search count  
# MAGIC   
# MAGIC Drawbacks to this approach:  
# MAGIC The URI counts do not include private browsing.  
# MAGIC The active hours do not include time watching videos or reading a long article.
# MAGIC 
# MAGIC The cutoff I've used below is **80th percentile for 1 week average of aDAU**: uri >= 169, active hours >= 0.93 hours
# MAGIC 
# MAGIC I checked this notebook for two additional sample ids for the same dates and the numbers were very similar.  
# MAGIC For sample id 42 the 80th percentile for 1 week average of aDAU: uri = 169.00, and active hours = 0.93 hours.  
# MAGIC For sample id 35 the 80th percentile for 1 week average of aDAU: uri = 168.71, and active hours = 0.93 hours.   
# MAGIC For sample id 78 the 80th percentile for 1 week average of aDAU: uri = 169.00, and active hours = 0.93 hours. 
# MAGIC 
# MAGIC I checked this notebook for a week in October and a week in January and the numbers increased in each week.  
# MAGIC For October the 80th percentile for 1 week average of aDAU: uri = 171.00, and active hours = 0.95 hours.  
# MAGIC For January the 80th percentile for 1 week average of aDAU: uri = 172.33, and active hours = 0.97 hours.  
# MAGIC **For this reason, heavy users cutoffs will need to be a methodology and not a fixed number.**

# COMMAND ----------

# MAGIC %md
# MAGIC #### Heavy Users - URI  
# MAGIC   
# MAGIC Clients with uri >= 169

# COMMAND ----------

# Heavy Users based on uri - one week
heavy_uri = '169'
ms_1week_hu_uri = ms_1week.where('td_uri >= ' +  heavy_uri)
display(ms_1week_hu_uri['submission_date_s3', 'td_uri', 'td_active_hours'].describe())

# COMMAND ----------

def  print_percents(num_hu_clients, num_avg_hu_clients, num_day_hu_clients, type, description):
    print 'One Week:'
    print '{0:,.0f}'.format(num_all_clients), ' total clients'
    print '  {0:,.0f}'.format(num_hu_clients), 'heavy ' + type + ' clients any day of the week'
    print '   %0.2f'%((num_hu_clients *1.0/num_all_clients) * 100), '% of the clients have heavy ' + description
    print ''
    print 'One Week Average:'
    print '{0:,.0f}'.format(num_all_clients), ' total clients'
    print '  {0:,.0f}'.format(num_avg_hu_clients), 'heavy ' + type + ' clients averaged over the week'
    print '   %0.2f'%((num_avg_hu_clients *1.0/num_all_clients) * 100), '% of the clients have average heavy ' + description
    print 'This is less than 20% because the cutoff was based on aDAU, not all users'
    print ''
    print 'One Day:'
    print '{0:,.0f}'.format(num_wday_clients), ' total clients'
    print '  {0:,.0f}'.format(num_day_hu_clients), ' heavy ' + type + ' clients'
    print '   %0.2f'%((num_day_hu_clients *1.0/num_wday_clients) * 100), '% of the clients have heavy ' + description   
    return None
  

# COMMAND ----------

num_huuri_clients = ms_1week_hu_uri.select('client_id').distinct().count()
num_avg_huuri_clients = ms_1week_avg.where('avg_uri >= ' +  heavy_uri).select('client_id').distinct().count()
wday_huuri_clients = ms_1week_hu_uri.where(wday_cond)
num_wday_huuri_clients = wday_huuri_clients.count()

print_percents(num_huuri_clients, num_avg_huuri_clients, num_wday_huuri_clients, 'URI', 'URI counts')

# COMMAND ----------

# Cut off these records at 7000 URI to generate a nice graph
ex_uri = '7000'
ms_1week_xhuuri = ms_1week.where('td_uri > ' + ex_uri)
print ms_1week_xhuuri.count(), 'records above 7000 uri for', ms_1week_xhuuri.select('client_id').distinct().count(), 'clients'
print 'The max value for uri is', '{0:,.0f}'.format(ms_1week_xhuuri.agg({'td_uri' : 'max'}).collect()[0][0])

# COMMAND ----------

# Look at the week's heavy user records
display(ms_1week.where('td_uri >= ' + heavy_uri + ' and td_uri <= ' + ex_uri).sort('client_id', 'submission_date_s3').select(ms_1week_cols))

# COMMAND ----------

# Generate histogram 
ms_1week_avg_huuri = ms_1week_avg.where('avg_uri >= ' + heavy_uri + ' and avg_uri <= ' + ex_uri)
huuri_avg_arry = array(ms_1week_avg_huuri.select('avg_uri').rdd.flatMap(lambda x: x).collect())

ms_week_huuri_max = ms_1week.where('td_uri >= ' + heavy_uri + ' and td_uri <= ' + ex_uri) \
          .groupby('client_id').agg(F.max('td_uri')).withColumnRenamed('max(td_uri)', 'max_td_uri').sort('client_id')
huuri_max_arry = array(ms_week_huuri_max.select('max_td_uri').rdd.flatMap(lambda x: x).collect())

ms_wday_huuri = ms_wday.where('td_uri >= ' + heavy_uri + ' and td_uri <= ' + ex_uri)
huuri_day_arry = array(ms_wday_huuri.select('td_uri').rdd.flatMap(lambda x: x).collect())

num_bins = 400
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the histogram of the data
#n, bins, patches = ax.hist(huuri_max_arry, num_bins, label = 'Week Max', color = 'slateblue')
ax.hist(huuri_avg_arry, num_bins, label = 'Week Avg', color = 'dodgerblue')
ax.hist(huuri_day_arry, num_bins, label = '1 Day', color = 'lightgreen')

ax.set_xlabel('URI Count')
ax.set_ylabel('Number of Clients')
ax.set_title(r'URI Distribution for Heavy Users')
ax.set_xlim(0, 1750)
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Heavy Users - Search Count

# COMMAND ----------

# Heavy Users based on search count - one week
#ms_1week_hu_sc = ms_1week.where('td_search_counts >= ' + heavy_sc)
#display(ms_1week_hu_sc['submission_date_s3', 'td_uri', 'td_active_hours', 'td_search_counts'].describe())

# COMMAND ----------

#num_husc_clients = ms_1week_hu_sc.select('client_id').distinct().count()
#num_avg_husc_clients = ms_1week_avg.where('avg_search_counts >= ' +  heavy_sc).select('client_id').distinct().count()
#wday_husc_clients = ms_1week_hu_sc.where(wday_cond)
#num_wday_husc_clients = wday_husc_clients.count()

#print_percents(num_husc_clients, num_avg_husc_clients, num_wday_husc_clients, 'search count', 'search counts')

# COMMAND ----------

# Cut off these records at 1000 search counts to generate a nice graph
#ms_1week_xhusc = ms_1week.where('td_search_counts > ' + ex_sc)
#print ms_1week_xhusc.count(), 'records above 1000 search counts for', ms_1week_xhusc.select('client_id').distinct().count(), 'clients'
#print 'The max value for search counts is', '{0:,.0f}'.format(ms_1week_xhusc.agg({'td_search_counts' : 'max'}).collect()[0][0])

# COMMAND ----------

# Generate histogram
#ms_1week_avg_husc = ms_1week_avg.where('avg_search_counts >= ' + heavy_sc + ' and avg_search_counts <= ' + ex_sc)
#husc_avg_arry = array(ms_1week_avg_husc.select('avg_search_counts').rdd.flatMap(lambda x: x).collect())
#ms_week_husc_max = ms_1week.where('td_search_counts >= ' + heavy_sc + ' and td_search_counts <= ' + ex_sc) \
#          .groupby('client_id').agg(F.max('td_search_counts')).withColumnRenamed('max(td_search_counts)', 'max_td_search_counts') \
#          .sort('client_id')
#husc_max_arry = array(ms_week_husc_max.select('max_td_search_counts').rdd.flatMap(lambda x: x).collect())
#ms_wday_husc = ms_wday.where('td_search_counts >= ' + heavy_sc + ' and td_search_counts <= ' + ex_sc)
#husc_day_arry = array(ms_wday_husc.select('td_search_counts').rdd.flatMap(lambda x: x).collect())
#
#num_bins = 1000
#plt.gcf().clear()
#fig, ax = plt.subplots(figsize=(10,6))
#
## the histogram of the data
##ax.hist(husc_max_arry, num_bins, label = 'Week Max', color = 'slateblue')
#ax.hist(husc_avg_arry, num_bins, label = 'Week Avg', color = 'dodgerblue')
#ax.hist(husc_day_arry, num_bins, label = '1 Day', color = 'lightgreen')
#
#ax.set_xlabel('Search Count')
#ax.set_ylabel('Number of Clients')
#ax.set_title(r'Search Count Distribution for Heavy Users')
#ax.legend(loc='best')
#
#display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Heavy Users - Active Hours  
# MAGIC   
# MAGIC Clients with active hours >= 0.93 hours

# COMMAND ----------

# Heavy Users based on active hours - one week
heavy_ah = '0.93'
ms_1week_hu_ah = ms_1week.where('td_active_hours >= ' + heavy_ah)
display(ms_1week_hu_ah['submission_date_s3', 'td_uri', 'td_active_hours'].describe())

# COMMAND ----------

num_huah_clients = ms_1week_hu_ah.select('client_id').distinct().count()
num_avg_huah_clients = ms_1week_avg.where('avg_active_hours >= ' +  heavy_ah).select('client_id').distinct().count()
wday_huah_clients = ms_1week_hu_ah.where(wday_cond)
num_wday_huah_clients = wday_huah_clients.count()

print_percents(num_huah_clients, num_avg_huah_clients, num_wday_huah_clients, 'active hour', 'active hours')

# COMMAND ----------

# Histogram of active hours in heavy range
ms_1week_avg_huah = ms_1week_avg.where('avg_active_hours >= ' + heavy_ah )
huah_avg_arry = array(ms_1week_avg_huah.select('avg_active_hours').rdd.flatMap(lambda x: x).collect())
ms_week_huah_max = ms_1week.where('td_active_hours >= ' + heavy_ah) \
          .groupby('client_id').agg(F.max('td_active_hours')).withColumnRenamed('max(td_active_hours)', 'max_td_active_hours') \
          .sort('client_id')
huah_max_arry = array(ms_week_huah_max.select('max_td_active_hours').rdd.flatMap(lambda x: x).collect())
ms_wday_huah = ms_wday.where('td_active_hours >= ' + heavy_ah)
huah_day_arry = array(ms_wday_huah.select('td_active_hours').rdd.flatMap(lambda x: x).collect())

num_bins = 3000
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the histogram of the data
#ax.hist(huah_max_arry, num_bins, label = 'Week Max', color = 'slateblue')
ax.hist(huah_avg_arry, num_bins, label = 'Week Avg', color = 'dodgerblue')
ax.hist(huah_day_arry, num_bins, label = '1 Day', color = 'lightgreen')

ax.set_xlabel('Active Hours')
ax.set_ylabel('Number of Clients')
ax.set_title(r'Active Hours Distribution for Heavy Users')
ax.set_xlim(0, 10)
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Heavy Users - Any of the 3 variables
# MAGIC 
# MAGIC Clients with uri >= 169, or active hours >= 0.93 hours

# COMMAND ----------

# Heavy Users, 80th percentile, 1 week aDAU
ms_1week_hu = ms_1week.where('td_uri >= ' +  heavy_uri + ' or td_active_hours >= ' + heavy_ah)
display(ms_1week_hu.select(ms_1week_cols).describe())

# COMMAND ----------

# Heavy User sample days
display(ms_1week_hu['submission_date_s3', 'td_uri', 'td_active_hours'])

# COMMAND ----------

num_hu_clients = ms_1week_hu.select('client_id').distinct().count()
num_avg_hu_clients = ms_1week_avg \
    .where('avg_uri >= ' +  heavy_uri + ' or avg_active_hours >= ' + heavy_ah) \
    .select('client_id').distinct().count()
num_wday_hu_clients = ms_1week_hu.where(wday_cond).count()

print_percents(num_hu_clients, num_avg_hu_clients, num_wday_hu_clients, 'use', 'usage in any category')

# COMMAND ----------

# MAGIC %md
# MAGIC Correlation of Heavy User Types

# COMMAND ----------

# three scatter plots: uri - sc, uri - ah, sc - ah
wday_hu = ms_1week_hu.where(wday_cond)
active_hrs = np.array(wday_hu.rdd.map(lambda p: p.td_active_hours).collect())
uri = np.array(wday_hu.rdd.map(lambda p: p.td_uri).collect())
search_counts = np.array(wday_hu.rdd.map(lambda p: p.td_search_counts).collect())

plt.gcf().clear()
fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))

#plot_scatter_line(ax1, search_counts, 'Search Counts', 1200, uri, 'URI', 2000)
plot_scatter_line(ax2, active_hrs, 'Active Hours', 20, uri, 'URI', 2000)
#plot_scatter_line(ax3, active_hrs, 'Active Hours', 20, search_counts, 'Search Counts', 1200)

plt.tight_layout()
display(fig)

# COMMAND ----------

print 'Correlation of search counts to uri is', wday_hu.stat.corr('td_search_counts', 'td_uri')
print 'Correlation of active hours to uri is', wday_hu.stat.corr('td_active_hours', 'td_uri')
print 'Correlation of active hours to search counts is', wday_hu.stat.corr('td_active_hours', 'td_search_counts')

# COMMAND ----------

# Get number of clients who are in both types of heavy users
#num_uri_sc = wday_huuri_clients.join(wday_husc_clients, 'client_id', 'inner').count()
num_uri_ah = wday_huuri_clients.join(wday_huah_clients, 'client_id', 'inner').count()
#num_sc_ah = wday_husc_clients.join(wday_huah_clients, 'client_id', 'inner').count()

# COMMAND ----------

# bar chart with three bars - all URI heavy users, URI heavy users also SC HU, URI heavy users also AH heavy users
# array for x axis
type_hu = ['URI', 'Active Hours']
# array for y axis, number of clients
num_clients_huuri = [num_wday_huuri_clients, num_uri_ah]
#num_clients_husc = [num_uri_sc, num_wday_husc_clients, num_sc_ah]
num_clients_huah = [num_uri_ah, num_wday_huah_clients]

width = 0.25
plt.gcf().clear()
fig, ax = plt.subplots(figsize=(10,6))

# the bar chart of the data
xticks = array(list(range(0, len(type_hu))))
ax.bar(xticks - width/2, num_clients_huuri, width, label = 'Heavy URI Clients', color = 'deepskyblue')
#ax.bar(xticks, num_clients_husc, width, label = 'Heavy Search Count Clients', color = 'lightsalmon')
ax.bar(xticks + width/2, num_clients_huah, width, label = 'Heavy Active Hour Clients', color = 'mediumaquamarine')

ax.set_xticks(xticks)
ax.set_xticklabels(type_hu)
ax.set_ylim(0, 250000)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)  
#ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Clients')
ax.set_xlabel('Also Heavy In', fontsize=13)
ax.xaxis.labelpad = 10 
ax.set_title(r'Number of Heavy Users By Type')
ax.legend(loc='best')

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Heavy Users - All of the 3 variables
# MAGIC 
# MAGIC Clients with uri >= 169 and active hours >= 0.93 hours

# COMMAND ----------

# Heavy Users, 80th percentile, 1 week aDAU
ms_1week_allhu = ms_1week.where('td_uri >= ' +  heavy_uri + ' and td_active_hours >= ' + heavy_ah)
display(ms_1week_allhu.select(ms_1week_cols).describe())

# COMMAND ----------

num_allhu_clients = ms_1week_allhu.select('client_id').distinct().count()
num_avg_allhu_clients = ms_1week_avg \
        .where('avg_uri >= ' +  heavy_uri + ' and avg_active_hours >= ' + heavy_ah) \
        .select('client_id').distinct().count()
num_wday_allhu_clients = ms_1week_allhu.where(wday_cond).count()

print_percents(num_allhu_clients, num_avg_allhu_clients, num_wday_allhu_clients, 'use', 'usage in all categories')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Heavy Users over Time

# COMMAND ----------

# MAGIC %md
# MAGIC Heavy Users over time can be seen in this notebook: <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/81242/command/81330">Heavy Users over Time</a>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attributes of Heavy Users

# COMMAND ----------

# MAGIC %md
# MAGIC The attributes of Heavy Users can be seen in this notebook: <a href="https://dbc-caf9527b-e073.cloud.databricks.com/#notebook/73994/command/76992">Heavy User Attributes</a>

# COMMAND ----------


