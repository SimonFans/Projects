import numpy
import random
from time import time
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

from matplotlib import cm
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


warehouse_location = abspath('/user/xxx/xxx_hiveDB.db')

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

spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict") 


class finaloutput:
    def __init__(self,event_date):
        self.event_date=event_date

    def create_output(self):
        final_output = """
        insert overwrite table push_analytics.promotions_recommendation
        partition (event_date)
        select consumer_id, bcookie,
            if(
               conv(
                  substr(
                      md5(concat(consumer_id, '-','2020v3runf')),
                      1, 6),
                  16,10)/conv('ffffff',16,10) > 0.90, 'hold out', best_offer) as best_offer,
                   next_best_offer,treatment, scored, event_date
            from 
            ( select
        a.consumer_id
        ,bcookie
        ,case when b.consumer_id is not null then grt_l3_cat_name
        when c.consumer_id is not null then grt_l2_cat_name
        else best_offer end as best_offer
        ,next_best_offer
        ,treatment
        ,scored
        ,"{0}" as event_date 
        from push_analytics.promo_result_final a
        left join 
        (select consumer_id, grt_l3_cat_name from
        push_analytics.promo_micro_v2_l3) b
        on a.consumer_id=b.consumer_id 
        left join 
        (select consumer_id, grt_l2_cat_name from
        push_analytics.promo_micro_v2_l2) c
        on a.consumer_id=c.consumer_id
        where scored='scored') temp
        where scored='scored'
        """.format(self.event_date)
        spark.sql(final_output)

event_date=sys.argv[1]
final = finaloutput(event_date=event_date)
print('pass line 1')
final.create_output()
print('Finished')
spark.stop()
