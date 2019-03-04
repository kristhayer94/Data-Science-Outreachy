# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will be used to determine if I should be using Active Hours or Subsesion Hours for my Heavy User analysis.

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as st
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Data

# COMMAND ----------

#Global variables
sample_id = 42
week_1_start = '20180923'
week_1_end = '20180929'
day_1_date = '20180926'

# COMMAND ----------

ping_query = """
    SELECT
        client_id,
        coalesce(scalar_parent_browser_engagement_total_uri_count, 0) AS uri_count,
        coalesce(scalar_parent_browser_engagement_active_ticks, 0) AS active_ticks,
        (coalesce(scalar_parent_browser_engagement_active_ticks, 0))*5/3600 AS active_hours,        
        subsession_length,
        (subsession_length/3600) AS subsession_hours,
        session_length,
        profile_subsession_counter,
        subsession_counter,
        submission_date_s3,
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
        active_experiment_branch,
        timezone_offset,
        vendor,
        is_default_browser,
        default_search_engine,
        devtools_toolbox_opened_count,
        client_submission_date,
        places_bookmarks_count,
        places_pages_count,
        scalar_content_telemetry_event_counts AS telem_event_counts,
        scalar_parent_browser_engagement_tab_open_event_count AS tab_event_count,
        scalar_parent_browser_engagement_window_open_event_count AS window_event_count,
        scalar_parent_browser_errors_collected_count AS errors_collected_count,
        scalar_parent_devtools_current_theme AS current_theme,
        scalar_parent_formautofill_availability AS formautofill_availability,
        scalar_parent_media_page_count AS media_page_count, 
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
        sum(scalar_parent_browser_engagement_tab_open_event_count) as td_tab_open_event_count,
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

# COMMAND ----------

ms_1week_ping = spark.sql(ping_query.format(week_1_start, week_1_end, sample_id))
ms_1week_sum = spark.sql(sum_query.format(week_1_start,week_1_end,sample_id))

# COMMAND ----------

ms_1week_avg = ms_1week_sum.groupBy('client_id').avg() \
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
    .withColumnRenamed('avg(sync_count_mobile)', 'avg_sync_mobile')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Correlation

# COMMAND ----------

# How correlated are the two hours values for pings?
ms_1week_ping.stat.corr('active_hours', 'subsession_hours')

#I also tested this for a week in January (Jan 6 - Jan 12) and got this correlation: 
# 0.002202641498433819

# COMMAND ----------

# How correlated are the two hours values for days?
ms_1week_sum.stat.corr('td_active_hours', 'td_subsession_hours')

#I also tested this for a week in January (Jan 6 - Jan 12) and got this correlation: 
# 0.014185426244677554

# COMMAND ----------

# How correlated are the two hours values averaged over a week?
ms_1week_avg.stat.corr('avg_active_hours', 'avg_subsession_hours')

# I also tested this for a week in January (Jan 6 - Jan 12) and got this correlation: 
# 0.00991343326194278

# COMMAND ----------

# Scatter plot of active hours and subsession hours for all pings
active_hrs = np.array(ms_1week_ping.rdd.map(lambda p: p.active_hours).collect())
subsession_hrs = np.array(ms_1week_ping.rdd.map(lambda p: p.subsession_hours).collect())
plt.gcf().clear()
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_title('Hours from Pings')
ax1.set_xlabel('Active Hours')
ax1.set_ylabel('Subsession Hours')
top = 30
bottom = 0
ax1.set_ylim(bottom, top)
ax1.set_xlim(0, 30)
plt.scatter(active_hrs, subsession_hrs, s=1)
display(fig)

# The scatter plot for the week in January has this same shape.

# COMMAND ----------

# MAGIC %md
# MAGIC The two hours values are not very correlated based on pings, daily totals or weekly averages.   
# MAGIC   
# MAGIC I could use both, or I could pick the one that I think is more reasonable based on the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Check  
# MAGIC   
# MAGIC The maximum active hours and subsession hours should be 24 hours (25 with latency) since a ping should be sent at least daily.

# COMMAND ----------

# MAGIC %md
# MAGIC For a week's worth of individual pings, the max active hours are a bit high, but the max subsession hours are very high.

# COMMAND ----------

# Look at the summary stats for ping values
display(ms_1week_ping['uri_count', 'active_ticks', 'active_hours', 'subsession_hours'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC There are only 4 pings in a week with active hours > 24 hours, and they are all for the same client_id.

# COMMAND ----------

ms_1week_ping_cols = ms_1week_ping.columns
ms_1week_ping_cols.remove('client_id')
# Look at the pings with high active hours
ms_1week_highat = ms_1week_ping.filter('active_hours > 24')
display(ms_1week_highat.select(ms_1week_ping_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC There are 198,466 pings with subsession hours > 25 hours in one week, representing 131,870, unique client_ids.  
# MAGIC 
# MAGIC The subsession hours measurement looks more suspect.  We could cap the pings at 25 hours, filter out the pings or the clients with high subsession hours.  Capping the pings would maintain the information if more than one ping is submitted on the same day or if multiple users are using the same client_id.

# COMMAND ----------

# Look at the summary stats for pings with high subsession hours
ms_1week_highss = ms_1week_ping.filter('subsession_hours > 25')
display(ms_1week_highss['uri_count', 'active_ticks', 'active_hours', 'subsession_hours'].describe())

# COMMAND ----------

# Look at the pings with high subsession hours
display(ms_1week_highss.select(ms_1week_ping_cols))

# COMMAND ----------

# How many distinct clients have high subsession hours?
ms_1week_highss.select('client_id').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC 
# MAGIC Active Hours 
# MAGIC * indicate when someone is actively using the browser
# MAGIC * will undercount times when watching Netflix or reading a long article
# MAGIC * data has a small standard deviation but the max values look reasonable
# MAGIC * only has one client id with data above the reasonable max
# MAGIC 
# MAGIC Subsession Hours
# MAGIC * will capture times when passively using the browser
# MAGIC * data has a very large standard deviation
# MAGIC * there are 131,870 client ids with data above the reasonable max
# MAGIC 
# MAGIC Based on all of these factors, I will use Active Hours in my Heavy User analysis.

# COMMAND ----------


