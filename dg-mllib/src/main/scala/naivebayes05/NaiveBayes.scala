package naivebayes05

import org.apache.spark.ml.Model
import org.apache.spark.mllib.classification.{NaiveBayes, LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.feature.{PCA, StandardScaler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable.ArrayBuffer
import scala.math.BigDecimal.RoundingMode


/**
 * Created by zyong on 2016/7/1.
 */
case class InstallRecord(uploadDate: String, uid: String, app: String)
case class UserTag(uid: String, dt: String, tag: Int)
case class TestResult(precise: Double, iteration: Int, bestStep: Double, batchFraction: Double)

object NaiveBayesApp {

  val conf = new SparkConf().setAppName("NaiveBayes").setMaster("local[2]")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._

  def main(args: Array[String]) {

    val basePath = "D:\\Research\\BI@Home\\SparkMLlib\\01RDD基础\\data1\\"
    //val basePath = "hdfs://ns/user/bifin/zyong/ml/data/"
    val installLog = "install_log"
    val userTag = "user_tag"

    val installLogRdd = sc.textFile(basePath + installLog).map(_.split("\\s")).map(
      ar => InstallRecord(ar(0), ar(1), ar(2))
    ).cache()

    val uidCnt = installLogRdd.map(i => (i.uid, 1))

    installLogRdd.map(_.app).distinct()

    val userTagRdd = sc.textFile(basePath + userTag).map(_.split("\\s")).map(
      ar => UserTag(ar(0), ar(1),
        if (ar(2).toDouble < 0.5) 0
        else if(ar(2).toDouble <= 1) 1
        else 2
      )
    ).cache()

    val dt = "2016-03-29"
    val data = generateData(dt, installLogRdd, userTagRdd)

    // 样本数据划分
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val (training, test) = (splits(0), splits(1))
    training.setName("Training").cache()

    //runLogisticRegression(training)

    val resultList = runNaiveBayes(training, test)
    resultList.foreach {
      rst =>
        val metrics = rst.metric
        println(
          s"""=============== ModelType: ${rst.model}, Lambda: ${rst.lambda} =================
             |${metrics.precision}
             |${metrics.recall}
           """.stripMargin)
        metrics.confusionMatrix
    }
    // 清理内存
    training.unpersist()


  }

  /**
   * 生产训练LabelPoints RDD
   * @param dt 数据日期
   * @param installLogRdd 安装记录
   * @param userTagRdd 用户标签
   * @return
   */
  def generateData(dt: String, installLogRdd: RDD[InstallRecord], userTagRdd: RDD[UserTag]): RDD[LabeledPoint] = {
    val ut = userTagRdd.filter(_.dt.equals(dt)).keyBy(_.uid).cache()
    val topk = installLogRdd.filter(_.uploadDate.equals(dt))
      .map(i => (i.app, 1)).reduceByKey(_ + _)
      .takeOrdered(20)(Ordering[Int].reverse.on(x=>x._2)).map(_._1)
    // 剔出前20安装的APP
    val install = installLogRdd.filter( i => i.uploadDate.equals(dt) && !topk.contains(i.app) ).keyBy(_.uid).cache()

    val data = install.groupByKey().map { case (uid, recordIterator) =>
      val records = recordIterator.toArray
      val installCount = records.length
      val installApps = records.map(_.app).distinct.length
      (uid, installCount, installApps)
    }.keyBy(_._1).join(ut).map { case (uid, (t, userTag)) =>
      LabeledPoint(userTag.tag, Vectors.dense(Array(t._2.toDouble))) // 选取用户安装次数单变量作为特征
      LabeledPoint(userTag.tag, Vectors.dense(Array(t._2.toDouble, t._3.toDouble))) // 选取安装次数和App种类数作为特征
    }

    // 数据标准化API调用
    //val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(_.features))
    //val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    install.unpersist()
    ut.unpersist()

    //scaledData
    data

  }

  def generateData2(dt: String, installLogRdd: RDD[InstallRecord], userTagRdd: RDD[UserTag]): RDD[LabeledPoint] = {
    val ut = userTagRdd.filter(_.dt.equals(dt)).keyBy(_.uid)
    val installOfDt = installLogRdd.filter(_.uploadDate.equals(dt))
    val appMap = installOfDt.map(_.app).distinct.collect.sorted.zipWithIndex.toMap

    //val appMapBC = ut.context.broadcast(appMap)
    installOfDt.groupBy(_.uid).map {
      case (uid, itr) =>
        //val appMap = appMapBC.value
        val records = itr.toArray
        val appFeature = Array.ofDim[Double](appMap.size)
        records.foreach {
          ir =>
            appMap.get(ir.app) match {
              case Some(idx) =>  appFeature(idx) = 1.0
              case _ =>
            }
        }
        (uid, appFeature)
    }.keyBy(_._1).join(ut).map {
      case (uid, ((_, appFeature), userTag)) =>
        LabeledPoint(userTag.tag, Vectors.dense(appFeature))
    }
  }

  case class NBTestResult(lambda: Double, model: String, metric: MulticlassMetrics)

  /**
   * 朴素贝叶斯
   * @param trainingData
   * @param testData
   * @return
   */
  def runNaiveBayes(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): ArrayBuffer[NBTestResult] = {
    var (bestLambda, bestModel) = (0, "")
    val resultList: ArrayBuffer[NBTestResult] = ArrayBuffer()
    for (
      lambda <- Seq(0.0, 0.25, 0.5, 0.75, 1.0);
      modelType <- Seq("multinomial")
    ) {
      val model = NaiveBayes.train(trainingData, lambda, modelType)
      val predictAndLabel = testData.map {
        lp =>
          (model.predict(lp.features), lp.label)
      }
      resultList += NBTestResult(lambda, modelType, new MulticlassMetrics(predictAndLabel))
    }
    resultList
  }

}

