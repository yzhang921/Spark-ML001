package fpgrowth08

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by zyong on 2016/8/17.
 */
object FPGrowthApp {
  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    conf.set("spark.kryo.classesToRegister", "scala.collection.mutable.ArrayBuffer,scala.collection.mutable.ListBuffer")
    val sc = new SparkContext(conf)

    val path = "D:\\Research\\BI@Home\\SparkMLlib\\01RDD基础\\data1\\install_log"
    //val path = "hdfs://ns/user/bifin/zyong/ml/data/install_log"
    //val path = "/data20/Download/sparkmllib-class/data1"

    val installRawData = sc.textFile(path)

    case class InstallRecord(uploadDate: String, uid: String, app: String)
    val installRecords = installRawData.map(_.split("\t")).map {
      case t => InstallRecord(t(0), t(1), t(2))
    }


    val oneDayData = installRecords.filter(_.uploadDate.equals("2016-03-26"))
    oneDayData.setName("oneDayData")
    oneDayData.persist(StorageLevel.MEMORY_AND_DISK_SER)
    oneDayData.count() // 221217

    val top100App = oneDayData.map(x => (x.app, 1)).reduceByKey(_ + _).takeOrdered(100)(Ordering[Int].reverse.on(_._2))

    val transaction = oneDayData.filter(x => top100App.map(_._1).contains(x.app)).groupBy(_.uid).map {
      case (uid, itr) =>
        itr.toArray.map(_.app).distinct
    }.cache()
    val fpg = new FPGrowth().setMinSupport(0.2).setNumPartitions(10)
    val model = fpg.run(transaction)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent.mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }
}
