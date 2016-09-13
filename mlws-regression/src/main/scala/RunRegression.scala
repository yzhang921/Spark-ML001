import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import math._
/**
 * Created by zyong on 2015/11/30.
 */
object RunRegression {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val path = "D:/Research/BI@Home/SparkClass/MachineLearningWithSpark/data/Bike-Sharing-Dataset/hour.csv"
    // val path = "/data20/workspace/sparkml/Bike-Sharing-Dataset/hour.csv"
    val raw_data: RDD[String] = sc.textFile(path).filter(!_.startsWith("instant"))
    val num_data = raw_data.count()
    val records: RDD[Array[String]] = raw_data.map(_.split(","))


    println("Number of Records: " + records.count())

    val cm = get_mapping(records, 2)
    val mappings: Seq[Map[String, Int]]= (2 to 9).map(idx => get_mapping(records, idx))

    val data: RDD[LabeledPoint] = records.map { r =>
      val features = extract_features(r, mappings)
      LabeledPoint(r.last.toDouble, Vectors.dense(features))
    }

    records.cache()
    runLinearRegression(data)
  }

  /**
   * 获取category特征map
   * @param rdd
   * @param idx
   * @return
   */
  def get_mapping(rdd: RDD[Array[String]], idx: Int): Map[String, Int] = {
    rdd.map(ar => ar(idx)).distinct.collect.sorted.zipWithIndex.toMap
  }

  /**
   * 规范化特征向量
   * @param record
   * @return
   */
  def extract_features(record: Array[String], mappings: Seq[Map[String, Int]]): Array[Double] = {
    var category_features: Array[Double] = Array()
    // 把向量的2到9位规范化为1-k
    (2 to 9).map { idx =>
      val mapping_idx = idx - 2 // mapping index from 0-7, need left switch 2 step
      val mapping = mappings(mapping_idx)
      val categoryNumber = mapping.keys.size
      val categoryFeature = Array.ofDim[Double](categoryNumber)
      categoryFeature(mapping(record(idx))) = 1.0
      category_features = category_features ++ categoryFeature
    }
    category_features ++ record.slice(10, record.size - 1).map(_.toDouble)
  }


  def runLinearRegression(data: RDD[LabeledPoint]): Unit = {
    val linear_model = LinearRegressionWithSGD.train(data, numIterations = 10, stepSize = 0.1)
    val true_vs_predicted = data.map(p => (p.label, linear_model.predict(p.features)))
    print("Linear Model predictions: ")

    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val dt_model = DecisionTree.trainRegressor(data, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val true_vs_dtpredicted = data.map(p => (p.label, dt_model.predict(p.features)))

    true_vs_dtpredicted.take(10).foreach(println _)

    val linemodel_mse = data.map(p => squared_error(p.label, linear_model.predict(p.features))).mean()
    val dtmodel_mse = data.map(p => squared_error(p.label, dt_model.predict(p.features))).mean()

    val linemodel_mae = data.map(p => abs_error(p.label, linear_model.predict(p.features))).mean()
    val dtmodel_mae = data.map(p => abs_error(p.label, dt_model.predict(p.features))).mean()

    val linemodel_rmsle = data.map(p => squared_log_error(p.label, linear_model.predict(p.features))).mean()
    val dtmodel_rmsle = data.map(p => squared_log_error(p.label, dt_model.predict(p.features))).mean()

    println(f"Linear Model - Mean Squared Error: ${linemodel_mse}%2.4f")
    println(f"DecisionTree Model - Mean Squared Error: ${dtmodel_mse}%2.4f")
    println
    println(f"Linear Model - Mean Squared Error: ${linemodel_mae}%2.4f")
    println(f"DecisionTree Model - Mean Squared Error: ${dtmodel_mae}%2.4f")
    println
    println(f"Linear Model - Mean Squared Error: ${linemodel_rmsle}%2.4f")
    println(f"DecisionTree Model - Mean Squared Error: ${dtmodel_rmsle}%2.4f")
  }

  /* 平方差 */
  def squared_error(actual: Double, predict: Double): Double = {
    pow((predict - actual), 2)
  }

  /*Mean Absolute Error*/
  def abs_error(actual: Double, predict: Double): Double = {
    abs(predict - actual)
  }

  /* Root Mean Squared Log Error */
  def squared_log_error(actual: Double, predict: Double): Double = {
    pow((log(predict + 1) - log(actual + 1)), 2)
  }

}
