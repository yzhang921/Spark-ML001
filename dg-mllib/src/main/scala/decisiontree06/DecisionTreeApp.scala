package decisiontree06

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.{StandardScaler, PCA}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.math.BigDecimal.RoundingMode
/**
 * Created by zyong on 2016/7/20.
 */
object DecisionTreeApp {

  case class Iris(sepalLength: Double , sepalWidth: Double, petalLength: Double, petalWidth: Double, species: String)
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    val basePath = "D:\\Research\\BI@Home\\SparkClass\\spark-1.5.1-bin-hadoop2.6\\data\\mllib"
    //val basePath = "hdfs://ns/user/bifin/zyong/ml/data/"
    val fileName = "iris.csv"

    // ID,Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species

    val iris = sc.textFile(basePath + fileName).map(_.split(",")).map(i => Iris(i(1).toDouble, i(2).toDouble, i(3).toDouble, i(4).toDouble, i(5)))
    iris.cache()
    iris.toDF().registerTempTable("iris")

    val data = iris.map(i =>
      LabeledPoint(
        i.species match {
          case "versicolor" => 0.0
          case "setosa" => 1.0
          case "virginica" => 2.0
        }, Vectors.dense(Array(i.sepalLength, i.sepalWidth, i.petalLength, i.petalWidth))))

    // 样本数据划分
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val (training, test) = (splits(0), splits(1))
    training.setName("Training").cache()

    val resultList = runDecisionTree(training, test)

    resultList.foreach{
      rst =>
        println(
          s"""=============== Impurity: ${rst.impurity}, MaxDepth: ${rst.maxDepth}, MaxBins: ${rst.maxBins} =================
             |${rst.metrics.precision}
             |${rst.metrics.recall}
           """.stripMargin)
        println(rst.metrics.confusionMatrix)
    }


    // 清理内存
    training.unpersist()
  }


  case class DCTestResult(impurity: String, maxDepth: Int, maxBins: Int, metrics: MulticlassMetrics)

  /**
   * 决策树训练
   * @param training
   * @param test
   * @return
   */
  def runDecisionTree(training: RDD[LabeledPoint], test: RDD[LabeledPoint]): ArrayBuffer[DCTestResult] = {
    val resultList: ArrayBuffer[DCTestResult] = ArrayBuffer()
    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]()
    /*
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    */

    for (
      impurity <- Seq("gini", "entropy");
      maxDepth <- Seq(2, 3);
      maxBins <- Seq(16, 32, 64)
    ) {
      val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
      val predictAndLabel = test.map( lp => (model.predict(lp.features), lp.label))

      model.toDebugString

      resultList += DCTestResult(impurity, maxDepth, maxBins, new MulticlassMetrics(predictAndLabel))
    }
    resultList
  }
}
