package recommd09

import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import recommd09.recommend.{ItemSimi, RecommendedItem, ItemSimilarity, ItemPref}

/**
 * Created by zyong on 2016/8/3.
 */
object RecommendApp {

  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val path = "D:\\Research\\BI@Home\\SparkMLlib\\01RDD基础\\data1\\install_log\\000000_0.gz"
    //val path = "hdfs://ns/user/bifin/zyong/ml/data/install_log"
    //val path = "/data20/Download/sparkmllib-class/data1"

    val installRawData = sc.textFile(path)

    case class InstallRecord(uploadDate: String, uid: String, app: String)
    val installRecords = installRawData.map(_.split("\t")).map{
      case t => InstallRecord(t(0), t(1), t(2))
    }


    val oneDayData = installRecords.filter(_.uploadDate.equals("2016-03-26"))
    oneDayData.setName("oneDayData")
    oneDayData.persist(StorageLevel.MEMORY_AND_DISK_SER)
    oneDayData.count() // 221217

    val top100App = oneDayData.map( x => (x.app, 1)).reduceByKey(_ + _).takeOrdered(100)(Ordering[Int].reverse.on(_._2))

    val userData = oneDayData.filter(x => top100App.map(_._1).contains(x.app)).map(x => ItemPref(x.uid, x.app, 1))

    //建立模型
    val similRdd = new ItemSimilarity().Similarity(userData, "cooccurrence")
    val recommendRDD = new RecommendedItem().Recommend(similRdd, userData, 30)

    similRdd.take(20).foreach (i =>
      println(i.itemid1 + ", " + i.itemid2 + ", " + i.similar)
    )

    recommendRDD.cache()
    val users = recommendRDD.map(_.userid).distinct().takeSample(false, 100, 10L)
    recommendRDD.filter( i => users.contains(i.userid)).collect().foreach( rc =>
      println(s"uid: ${rc.userid}, Recommand: ${rc.itemid}, Pref: ${rc.pref}")
    )
  }

}
