package rddbase01

import org.apache.spark.mllib.fpm.{FPGrowthModel, FPGrowth}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.math.BigDecimal.RoundingMode

/**
 * Created by zyong on 2016/6/12.
 */
case class InstallRecord(uploadDate: String, uid: String, app: String)

object RddBase {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val path = "D:\\Research\\BI@Home\\SparkMLlib\\01RDD基础\\data1\\000000_0.gz"
    //val path = "hdfs://ns/user/bifin/zyong/ml/data"
    //val path = "/data20/Download/sparkmllib-class/data1"

    val installRawData = sc.textFile(path)

    val installRecords = installRawData.map(_.split("\t")).map{
      case t => InstallRecord(t(0), t(1), t(2))
    }

    //并且统计数据总行数、用户数量、日期有哪几天
    installRecords.setName("installRecords")
    installRecords.persist(StorageLevel.MEMORY_AND_DISK_SER)
    installRecords.count() // 6503485
    installRecords.map(_.uid).distinct().count() // 108238
    installRecords.map(_.uploadDate).distinct().count() // 26

    // 用户新装包统计
    val sortByDateCount = installRecords.map(i => (i.uploadDate, 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false)
    sortByDateCount.take(2).foreach(print)
    val newInstallByUser = newInstall(installRecords, "2016-03-29", "2016-03-30")

    newInstallByUser.take(10).foreach{ i =>
      val first3apps = i._2.take(3).mkString(";")
      println(s"User: ${i._1}, first 3 APPs: ${first3apps}")
    }

    // 频率分析
    // 指定任意1天的数据, 统计每个包名的安装用户数量，由大到小排序，并且取前1000个包名，最后计算这1000个包名之间的支持度和置信度
    val oneDayData = installRecords.filter(_.uploadDate.equals("2016-03-26"))
    oneDayData.setName("oneDayData")
    oneDayData.persist(StorageLevel.MEMORY_AND_DISK_SER)
    oneDayData.count() // 221217
    val top1kApps = oneDayData.map(x => (x.app, 1)).reduceByKey(_ + _).takeOrdered(1000)(Ordering[Int].reverse.on(x=>x._2))

// 包含top1000的安装子集
    val top1kAppsBC = sc.broadcast(top1kApps.map(_._1))
    val top1kRecords = oneDayData.filter{x: InstallRecord => top1kAppsBC.value.contains(x.app)}
    top1kRecords.setName("top1kRecords")
    top1kRecords.persist(StorageLevel.MEMORY_AND_DISK_SER)
    top1kRecords.count() //110506

    // 按用户分Transaction
    val txs = top1kRecords.groupBy(_.uid)
    txs.setName("Transactions")
    txs.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val txCnt = txs.count() // 6664

    val minSupport = 0.1
    val minConfidence = 0.3


    val uniq = top1kRecords.map(i => (i.uid, i.app)).distinct()
    val freq = uniq.join(uniq).map( i => (i._2, 1) ).reduceByKey(_ + _)
    freq.setName("Frequent Item Pairs")
    freq.persist(StorageLevel.MEMORY_AND_DISK)

    val support = freq.map {i =>
      val (pair, freq)  = (i._1, i._2)
      val supportValue = scala.math.BigDecimal(freq*1.0/txCnt).setScale(4, RoundingMode.UP)
      (pair._1, pair._2, freq, supportValue)
    }.filter(_._3 > minSupport)

    val top1kAppsCntBC = sc.broadcast(top1kApps)
    val confident = support.map { row =>
      val appCntDic = top1kAppsCntBC.value.toMap
      val (appA, appB, pairFreq, support) = (row._1, row._2, row._3, row._4)
      val a2b = scala.math.BigDecimal(pairFreq*1.0/appCntDic.get(appA).get).setScale(4, RoundingMode.UP)
      val b2a = scala.math.BigDecimal(pairFreq*1.0/appCntDic.get(appB).get).setScale(4, RoundingMode.UP)
      (appA, appB, pairFreq, support, a2b, b2a)
    }.filter( i => i._5 > minConfidence && i._6 > minConfidence)

    print(confident)
    installRecords.unpersist()
    oneDayData.unpersist()
    top1kRecords.unpersist()
    txs.unpersist()
  }

  def newInstall(installRecords: RDD[InstallRecord], date: String, date_next: String ) : RDD[(String, Iterable[String])] = {
    val apps = installRecords.filter(_.uploadDate.equals(date)).map(i => (i.uid, i.app))
    val next_apps = installRecords.filter(_.uploadDate.equals(date_next)).map(i => (i.uid, i.app))
    next_apps.subtract(apps).groupByKey()
  }


}
