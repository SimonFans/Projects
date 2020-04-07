import numpy
import random
from time import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import Row
from os.path import expanduser, join, abspath
from pyspark.sql.functions import col,asc,sum,avg,stddev,countDistinct,sqrt,variance,lit,concat
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql import types as t
from pyspark.sql.functions import lit
from pyspark.sql import DataFrameStatFunctions as statFunc

from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
#from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql.functions import col, asc
from pyspark.ml.feature import MinMaxScaler

from pyspark.sql import SparkSession
from pyspark.sql import Row
import sys
import getpass
from matplotlib import cm
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

user = getpass.getuser()

os.system("hadoop fs -rm -R -skipTrash /user/{0}/.Trash".format(user))

warehouse_location = abspath('/user/{0}/{0}_hiveDB.db').format(user)

spark = SparkSession \
.builder \
.master("yarn") \
.config("spark.app.name", "email_cadence_model_score_v2")\
.config("spark.driver.maxResultSize", "40G")\
.config("spark.driver.memory", "32G")\
.config("spark.dynamicAllocation.enabled", "false")\
.config("spark.executor.cores", 2)\
.config("spark.executor.instances", 25)\
.config("spark.executor.memory", "10G")\
.config("spark.kryoserializer.buffer.max", "1024M")\
.config("spark.network.timeout", "800s")\
.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
.config("spark.shuffle.service.enabled", "true")\
.config("spark.sql.hive.convertMetastoreOrc", "false")\
.config("spark.yarn.dist.files", "/var/groupon/spark-2.4.0/python/lib/pyspark.zip,/var/groupon/spark-2.4.0/python/lib/py4j-0.10.7-src.zip")\
.config("spark.yarn.dist.archives", "/var/groupon/spark-2.4.0/R/lib/sparkr.zip#sparkr,hdfs:////user/grp_gdoop_admin/anaconda/anaconda3_env.zip#ANACONDA")\
.config("spark.executorEnv.PYTHONPATH", "pyspark.zip:py4j-0.10.7-src.zip")\
.config("spark.executor.memoryOverhead", "8192")\
.config("spark.yarn.queue", "public")\
.config("spark.sql.warehouse.dir", warehouse_location)\
.enableHiveSupport() \
.getOrCreate()




class micro_targeting_l2_recommender():
    def __init__(self,event_date):
        self.event_date=event_date
        #pass

    def get_data(self):
        query="""select * 
        from push_analytics.deal_view_conversion 
        where event_date between date_sub(current_date,60) and date_sub(current_date,1) and select_flag=0
        and grt_l2_cat_name in (select index_name from  push_analytics.xm_index a  inner join  
        push_analytics.xm_tb1 b
        on a.promo_id = b.promo_id where promo_date = "{0}" and sub_category='L2')
        """.format(self.event_date)
        test_df=spark.sql(query)\
        .select(['grt_l2_cat_name','event_date','consumer_id','deal_uuid','deal_views','grt_l2_purchaser_14d','purchaser_14d','unique_deal_views'])
        
        return test_df
    
    def pre_process_data(self,raw_df):
        df=raw_df\
        .select(['grt_l2_cat_name','event_date','consumer_id','deal_uuid','deal_views','grt_l2_purchaser_14d','purchaser_14d','unique_deal_views'])\
        .groupby(['consumer_id','grt_l2_cat_name','event_date'])\
        .agg(F.sum(col('deal_views')).alias('udv')
            ,F.sum(col('unique_deal_views')).alias('dv')
            ,F.countDistinct(col('deal_uuid')).alias('ud')
            ,F.max(col('grt_l2_purchaser_14d')).alias('grt_l2_purchaser_14d')
            ,F.max(col('purchaser_14d')).alias('purchaesr_14d'))

        df.createOrReplaceTempView("df") 

        df_processed=spark.sql(
            """SELECT *,sum(dv) OVER ( PARTITION BY consumer_id,grt_l2_cat_name ORDER BY datediff(event_date,'2007-01-01') 
            RANGE BETWEEN  29  PRECEDING AND CURRENT ROW) AS dvs_p30days
            ,sum(udv) OVER ( PARTITION BY consumer_id,grt_l2_cat_name ORDER BY datediff(event_date,'2007-01-01') 
            RANGE BETWEEN  29 PRECEDING AND CURRENT ROW) AS udvs_p30days 
            ,sum(ud) OVER ( PARTITION BY consumer_id,grt_l2_cat_name ORDER BY datediff(event_date,'2007-01-01') 
            RANGE BETWEEN 29  PRECEDING AND CURRENT ROW) AS ud_p30days 
            ,row_number() OVER (PARTITION BY consumer_id,grt_l2_cat_name ORDER BY datediff(event_date,'2007-01-01') desc ) AS recent_event_date
             FROM df
            order by consumer_id""").filter(col("recent_event_date")==1)
        
        return df_processed
        
        
    def baseline_model(self,df_processed):
        cond_dvs=F.when(col("dvs_p30days")> 400,400).otherwise(col("dvs_p30days"))


        stats=df_processed\
        .withColumn("dvs_p30days",cond_dvs)\
        .filter(col("recent_event_date")==1)\
        .filter(col("event_date")>=F.date_add(F.current_date(),-15))\
        .groupby(['grt_l2_cat_name'])\
        .agg(F.avg(col("dvs_p30days")).alias("avg_dvs_p30days")
            ,F.stddev(col("dvs_p30days")).alias("std_deal_view")
            ,F.round(F.avg(col("dvs_p30days")).cast('integer')).alias("avg_deal_view")
            ,F.max("dvs_p30days").alias("max_deal_view"))


        w =  Window.partitionBy(F.col('consumer_id')).orderBy(F.col('normalized_dvs_p30days').desc())

        df_final=df_processed\
        .filter(col("recent_event_date")==1)\
        .filter(col("event_date")>=F.date_add(F.current_date(),-15))\
        .join(stats,on='grt_l2_cat_name')\
        .withColumn('normalized_dvs_p30days',(col('dvs_p30days')-col('avg_deal_view'))/col('std_deal_view'))\
        .withColumn('normalized_dvs_p30days_rank',F.row_number().over(w))
        
        
        df_micro=df_final\
        .filter(col('normalized_dvs_p30days')>=0)\
        .filter(col('normalized_dvs_p30days_rank')==1)\
        .filter(col('grt_l2_purchaser_14d')==0)
        
        return df_micro
    
    def create_ouptput(self,df_micro):
        df_micro.createOrReplaceTempView("resultdf")
        spark.sql("""drop table if exists  push_analytics.promo_micro_v2_l2 purge """)
        spark.sql("""create table push_analytics.promo_micro_v2_l2 stored as orc 
        TBLPROPERTIES ('orc.compress'='SNAPPY') as select 
        a.consumer_id
        ,a.grt_l2_cat_name
        ,if(
           conv(
              substr(
                  md5(concat(a.consumer_id, '-','2020microv1')),
                  1, 6),
              16,10)/conv('ffffff',16,10) > 0.50, 'treatment', 'control') as treatment
        ,"{0}" as event_date 
        from resultdf a""".format(self.event_date))
        

        
event_date=sys.argv[1]
micro_baseline_l2 = micro_targeting_l2_recommender(event_date=event_date)
print('pass the first line')
raw_df=micro_baseline_l2.get_data()
print('pass the second line')
df_processed=micro_baseline_l2.pre_process_data(raw_df)
print('pass the third line')
df_micro=micro_baseline_l2.baseline_model(df_processed)
print('pass the fourth line')
micro_baseline_l2.create_ouptput(df_micro)
print('done')
spark.stop()
