package logistic_regression04


import org.apache.spark.ml.Model
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.{PCA, StandardScaler}
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD}
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

object LogisticRegression {

  val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
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
      ar => UserTag(ar(0), ar(1), if (ar(2).toDouble > 0.5) 1 else 0)
    ).cache()

    installLogRdd.toDF().registerTempTable("installLog")
    userTagRdd.toDF().registerTempTable("uidTag")

    // 验证用户标签数据是否在某一天有重复
    sqlContext.sql(
      """SELECT uid, dt, count(*)
        |  FROM uidTag
        | GROUP BY uid, dt
        |HAVING count(*) > 1
        | Limit 10""".stripMargin).show()

    val dt = "2016-03-29"

    val data = generateData(dt, installLogRdd, userTagRdd)

    // 样本数据划分
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val (training, test) = (splits(0), splits(1))
    training.setName("Training").cache()

    runLogisticRegression(training, test)

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
    val install = installLogRdd.filter(_.uploadDate.equals(dt)).keyBy(_.uid).cache()
    val ut = userTagRdd.filter(_.dt.equals(dt)).keyBy(_.uid).cache()

    val data = install.groupByKey().map { case (uid, recordIterator) =>
      val records = recordIterator.toArray
      val installCount = records.length
      val installApps = records.map(_.app).distinct.length
      (uid, installCount, installApps)
    }.keyBy(_._1).join(ut).map { case (uid, (t, userTag)) =>
      //LabeledPoint(userTag.tag, Vectors.dense(Array(t._2.toDouble))) // 选取用户安装次数单变量作为特征
      LabeledPoint(userTag.tag, Vectors.dense(Array(t._2.toDouble, t._3.toDouble))) // 选取安装次数和App种类数作为特征
    }

    // 数据标准化API调用
    // val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(_.features))
    // val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    install.unpersist()
    ut.unpersist()

    // scaledData
    data

  }

  /**
   * 逻辑回归训练
   * @param trainingData
   */
  def runLogisticRegression(trainingData: RDD[LabeledPoint], test: RDD[LabeledPoint]) = {

    // 参数设计
    var (bestPrecise, bestIteration, bestStep, bestBatchFraction) = (0.0, 0, 0.0, 0.0)
    val resultList: ArrayBuffer[TestResult] = ArrayBuffer()
    val (numberIterations, stepSize, miniBatchFraction) = (20, 0.1, 1.0)

    for (
      numberIterations <- Seq(20);
      stepSize <- Seq(0.1);
      miniBatchFraction <- Seq(1.0)
    ) {
      val model = LogisticRegressionWithSGD.train(trainingData, numberIterations, stepSize, miniBatchFraction)
      // val model = new LogisticRegressionWithLBFGS(training, numberIterations, stepSize, miniBatchFraction)
      model.clearThreshold()
      model.predict(test.map(_.features)).take(10).map(
        scala.math.BigDecimal(_).setScale(4, RoundingMode.UP)
      )
      val v_precise = precise(test, model)

      resultList += TestResult(v_precise, numberIterations, stepSize, miniBatchFraction)

      if (bestPrecise < v_precise) {
        bestPrecise = v_precise
        bestIteration = numberIterations
        bestStep = stepSize
        bestBatchFraction = miniBatchFraction
      }
    }

    println(s"Best Training parameters: numberIterations: ${bestIteration}; stepSize: ${bestStep} ; miniBatchFraction: ${bestBatchFraction}, " +
      s"Precise: ${bestPrecise}")
    resultList.foreach(println _)
  }


  /**
   * 训练精度
   * @param test 测试数据集
   * @param model 模型
   * @return
   */
  def precise(test: RDD[LabeledPoint], model: LogisticRegressionModel): Double = {

    model.clearThreshold()
    // 测试数据集预测
    val predictAndLabels = test.map {
      case LabeledPoint(label, features) => (model.predict(features), label)
    }

    // 计算精度
    val metrics = new MulticlassMetrics(predictAndLabels)
    metrics.precision

  }


}
