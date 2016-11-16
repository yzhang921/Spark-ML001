package extract_transform_select

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zyong on 2016/11/16.
  */
class BasicContext {

  val conf = new SparkConf().setAppName("ETS").setMaster("local")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

}
