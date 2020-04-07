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


warehouse_location = abspath('/user/xx/xx_hiveDB.db')

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




class promo_recommender_baseline:
    def __init__(self,features,event_date):
        self.x_features=features
        self.event_date=event_date

    def get_train_data(self):
        cond_purchase=F.when(col("purchases_etm")>0,1).otherwise(0)
        cond_treatment=F.when(col("variant")==lit('control'),1).otherwise(0)
        cond_treat_flag=F.when(col("control")==1,0).otherwise(1)


        cond_platform=F.when(col("last_platform")==lit('ipad'),'app') \
        .when(col("last_platform")==lit('groupon_customer_service'),'web') \
        .when(col("last_platform")==lit('groupon_desktop_web'),'web') \
        .when(col("last_platform")==lit('android'),'app') \
        .when(col("last_platform")==lit('iphone'),'app') \
        .when(col("last_platform")==lit('other mobile'),'app') \
        .when(col("last_platform")==lit('None'),'app') \
        .when(col("last_platform").isNull(),'app') \
        .otherwise(col('last_platform'))


        train_query="""select a.*,
        case when variant='personalized_promotions_2019-08-13_control' then 'control'
        when variant='personalized_promotions_2019-08-13_20% Off Apparel' then '20% Off Apparel'
        when variant='personalized_promotions_2019-08-13_SWS (20/10/10)' then 'SWS'
        when variant='personalized_promotions_2019-08-13_15% Off Local' then '15% off Local'
        when variant='personalized_promotions_2019-08-13_10% Off Goods' then '10% off Goods'
        when variant='personalized_promotions_2019-08-13_20% Off Local' then '20% Off Local'
        when variant='personalized_promotions_2019-08-13_15% Off Local/30% Off TTD' then '15% Off Local-30% TTD'
        else variant end as variant_new
        from push_analytics.promotions_exp_consumer  a
        where 
        (start_date between '2019-04-16' and '2019-05-14')"""

        train_data=spark.sql(train_query)\
        .filter(~col('9block_segment').isin(['Reactivation','Acquisition']))\
        .filter(col('days_since_last_purchase')<=275) \
        .filter(col("select_flag")==0)\
        .drop('variant') \
        .withColumnRenamed("variant_new","variant")
        
        train_data.createOrReplaceTempView("train_data")
        
        train_df = spark.sql("""
        select *
        from train_data
        where variant in (select index_name from  push_analytics.xm_index a  inner join  
        push_analytics.xm_tb1 b
        on a.promo_id = b.promo_id where promo_date = "{0}" and category='Broad')
        """.format(self.event_date))
        
        
#         train_df.createOrReplaceTempView("train_df")
#         max_ogp = spark.sql("""select min(ogp_etm) num from (
#         select ogp_etm, PERCENT_RANK() over(order by ogp_etm) percentile from train_df
#         where ogp_etm >0) temp
#         where percentile > = 0.99
#         """).toPandas()
#         ogp_etm_max = max_ogp.iloc[0,0]

        
        ogp_etm_max=train_df \
        .filter(col('ogp_etm') > 0) \
        .approxQuantile('ogp_etm', [0.995],relativeError=0.0001)
#        ogp_etm_max = statFunc(train_df.filter(col('ogp_etm')>0)).approxQuantile( "ogp_etm", [0.995], 0.0001)



        print ("Outlier max for oGP {}".format(ogp_etm_max))


#        print ("Outlier max for oGP {}".format(str(ogp_etm_max[0])))

        cond_ogp=F.when(col("ogp_etm")> ogp_etm_max[0],ogp_etm_max[0]).otherwise(col("ogp_etm"))


        cond_od_usage_flag=F.when(col("od_orders_365")>0,1).otherwise(0)
        cond_email_click_p30_flag=F.when(col("email_clicks_p30")>0,1).otherwise(0)
        ##assign null to the most frequent category. Note this can done only for active users (atleast 1 purchase in the last 365 days)
        cond_affinity=F.when(col("affinity").isNull(),'TTD').otherwise(col('affinity'))


        train_df=train_df.withColumn("purchasers_etm",cond_purchase)\
        .withColumn("control",cond_treatment)\
        .withColumn("treatment",cond_treat_flag)

        train_df=train_df \
                .withColumn("ogp_etm",cond_ogp)\
                .withColumn("od_usage_p365_flag",cond_od_usage_flag)\
                .withColumn("email_click_p30_flag",cond_email_click_p30_flag)\
                .withColumn("affinity",cond_affinity)\
                .withColumn("last_platform",cond_platform)
        return train_df

    def train_baseline_model(self,train_df,persist_model=False):
        """
        arg: training dataframe 
        return: dataframe with the best offer for 
        """
        segment_var=["control"] + self.x_features + ["variant"]
        segment=self.x_features
        segment_score=self.x_features+ ["variant"]
        
        window_1=Window \
        .partitionBy(segment) \
        .orderBy(col("control").desc())

        window_gp=Window \
        .partitionBy(segment) \
        .orderBy(col("gp_diff").desc())

        window_purchaser=Window \
        .partitionBy(segment) \
        .orderBy(col("purchaser_diff").desc())

        window_spend=Window \
        .partitionBy(segment) \
        .orderBy(col("efficiency").desc())

        df_pandas=train_df \
        .groupby(segment_var) \
        .agg(sum('purchasers_etm').alias("sum_purchasers_sd") \
             ,sum('ogp_etm').alias("sum_ogp_sd") \
             ,sum('od_discount_amount_etm').alias("sum_od_spend_sd")
             ,countDistinct(*['consumer_id','start_date']).alias("users"))\
        .withColumn("sum_purchasers_sd_control",F.first(col("sum_purchasers_sd")).over(window_1)) \
        .withColumn("sum_ogp_sd_control",F.first(col("sum_ogp_sd")).over(window_1)) \
        .withColumn("sum_od_spend_sd_control",F.first(col("sum_od_spend_sd")).over(window_1)) \
        .withColumn("users_control",F.first(col("users")).over(window_1)) \
        .withColumn("gp_diff",col("sum_ogp_sd")/col("users")-col("sum_ogp_sd_control")/col("users_control")) \
        .withColumn("purchaser_diff",col("sum_purchasers_sd")/col("users")-col("sum_purchasers_sd_control")/col("users_control"))\
        .withColumn("spend_diff",col("sum_od_spend_sd")/col("users")-col("sum_od_spend_sd_control")/col("users_control"))\
        .withColumn("efficiency",col("gp_diff")/col("spend_diff")) \
        .filter(col('control')==0)\
        .withColumn("gp_rank",F.row_number().over(window_gp)) \
        .withColumn("purchaser_rank",F.row_number().over(window_purchaser)) \
        .withColumn("efficiency_rank",F.row_number().over(window_spend))
        
        

        model_df=df_pandas \
        .select(segment_score+['gp_diff','gp_rank','users']) 
        #.filter(col("gp_rank")==1)
        
        if persist_model:
            pass

        return model_df
    
    def get_scoring_data(self):
            
        cond_platform=F.when(col("last_platform")==lit('ipad'),'app') \
        .when(col("last_platform")==lit('groupon_customer_service'),'web') \
        .when(col("last_platform")==lit('groupon_desktop_web'),'web') \
        .when(col("last_platform")==lit('android'),'app') \
        .when(col("last_platform")==lit('iphone'),'app') \
        .when(col("last_platform")==lit('other mobile'),'app') \
        .when(col("last_platform")==lit('None'),'app') \
        .when(col("last_platform").isNull(),'app') \
        .otherwise(col('last_platform'))
        
        cond_od_usage_flag=F.when(col("od_orders_365")>0,1).otherwise(0)
        cond_email_click_p30_flag=F.when(col("email_clicks_p30")>0,1).otherwise(0)
        cond_affinity=F.when(col("affinity").isNull(),'TTD')\
        .when(col("affinity")==lit("None"),'TTD').otherwise(col('affinity'))
        
        score_query="""select * from push_analytics.promotions_consumer_daily 
        where record_date = date_sub(current_date,2) and brand='groupon' """
        
        score_df=spark.sql(score_query)\
        .withColumn("od_usage_p365_flag",cond_od_usage_flag)\
        .withColumn("email_click_p30_flag",cond_email_click_p30_flag)\
        .withColumn("affinity",cond_affinity)\
        .withColumn("last_platform",cond_platform)
        score_df.count()
        
        return score_df
    
    def score_baseline_model(self,score_df,model_df):
        
        scoring_fields=["consumer_id"]+["select_flag","days_since_last_purchase"]+self.x_features

        cond_offer=F.when(col("gp_diff")<0,'No offer').otherwise(col('best_offer'))

        next_cond_offer=F.when(col("gp_diff")<0,'No offer').otherwise(col('next_best_offer'))

        final_output=score_df.select(*scoring_fields) \
        .join(model_df.filter(col('gp_rank')==1),self.x_features,how='left')\
        .withColumnRenamed("variant","best_offer")\
        .withColumn("best_offer",cond_offer)\
        .drop('gp_diff')\
        .drop('gp_rank')\
        .drop('users')\
        .join(model_df.filter(col('gp_rank')==2),self.x_features,how='left')\
        .withColumnRenamed("variant","next_best_offer")\
        .withColumn("next_best_offer",next_cond_offer)\
        .drop('gp_diff')\
        .drop('gp_rank')\
        .drop('users')
        
        #final_output.select(F.countDistinct(col('consumer_id'))).show()

        return final_output
    
    def create_scoring_history(self,scored_df):
        
        scored_df.createOrReplaceTempView("resultdf")
        
        spark.sql("""drop table if exists push_analytics.promo_result_final purge """)
        
        spark.sql("""create table push_analytics.promo_result_final 
        stored as orc TBLPROPERTIES ('orc.compress'='SNAPPY') as select a.* 
        ,if(
           conv(
              substr(
                  md5(concat(a.consumer_id, '-','2020v1runf')),
                  1, 6),
              16,10)/conv('ffffff',16,10) > 0.50, 'treatment', 'control') as treatment
        ,case when 9block_segment in ('Reactivation','Acquisition') then 'Not scored' 
        when days_since_last_purchase>275  then 'Not scored' 
        when select_flag=1 then 'Not scored' else 'scored' end as scored
        ,b.bcookie
        ,"{0}" as event_date 
        from resultdf a 
        left join 
        (select consumer_id,bcookie 
        from (select user_uuid as consumer_id
        ,bcookie
        ,row_number() over(partition by user_uuid order by app_download_date desc,last_push_send_date) as row_rank 
        from prod_groupondw.mobile_push_notification 
        where user_uuid is not null and bcookie is not null) a
        where row_rank=1) b on a.consumer_id=b.consumer_id""".format(self.event_date))


event_date=sys.argv[1]
print(event_date)
#features=sys.argv[2]
features=["9block_segment","affinity","od_usage_p365_flag","email_click_p30_flag","last_platform"]
promo_recommender=promo_recommender_baseline(features=features,event_date=event_date)
print('pass line 1')
train_df=promo_recommender.get_train_data()
print('pass line 2')
model_df = promo_recommender.train_baseline_model(train_df)
print('pass line 3')
score_df = promo_recommender.get_scoring_data()
print('pass line 4')
scored_df=promo_recommender.score_baseline_model(score_df,model_df)
print('pass line 5')
promo_recommender.create_scoring_history(scored_df)
print('done')
spark.stop()

