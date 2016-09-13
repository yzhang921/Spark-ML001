package lineregresion03

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.mllib.feature.{PCA, StandardScaler}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors

/**
 * Created by zyong on 2016/6/28.
 */
object LinearRegression {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val basePath = "D:\\Research\\BI@Home\\SparkClass\\spark-1.5.1-bin-hadoop2.6\\data\\mllib"
    //val basePath = "file:///home/bifin/zyong/sparkml/data/mllib"
    val fileName = "ridge-data/lpsa.data"

    val records = sc.textFile(basePath + "/" + fileName)

    val labeledPoints = records.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }.cache()

    val vectors = labeledPoints.map(_.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()

    println(matrixSummary.mean)
    println(matrixSummary.min)
    println(matrixSummary.max)
    println(matrixSummary.variance)

    //val numIterations = 100
    //val stepSize = 1
    //val miniBatchFraction = 1.0

    var (bestRMSE, bestIteration, bestStep, bestBatchFraction) = (Double.MaxValue, 0, 0.0, 0.0)

    for (
         numberIterations <- Seq(100, 200, 500);
         stepSize <- Seq(0.1, 0.5, 1);
         miniBatchFraction <- Seq(1.0, 0.5)
    ) {
      println("*******************************************************************************************************************************")
      val model = LinearRegressionWithSGD.train(labeledPoints, numberIterations, stepSize, miniBatchFraction)
      println(
        s""" Training parameters: numberIterations: ${numberIterations}; stepSize: ${stepSize} ; miniBatchFraction: ${miniBatchFraction}""".stripMargin)
      val error = rmse(labeledPoints, model)
      println(s"RMSE= ${error}")
      println

      if (bestRMSE > error) {
        bestRMSE = error
        bestIteration = numberIterations
        bestStep = stepSize
        bestBatchFraction = miniBatchFraction
      }

    }

    println(s"Best Training parameters: numberIterations: ${bestIteration}; stepSize: ${bestStep} ; miniBatchFraction: ${bestBatchFraction}, " +
      s"Error: ${bestRMSE}")


    for (k <- List(2, 3, 4, 5, 6, 7)) {
      val pca = new PCA(k).fit(labeledPoints.map(_.features))
      val trainingPCA = labeledPoints.map(p => p.copy(features = pca.transform(p.features)))
      val bestModelAndPCAModel = LinearRegressionWithSGD.train(trainingPCA, bestIteration, bestStep, bestBatchFraction)
      val error = rmse(trainingPCA, bestModelAndPCAModel)
      println(s"PCA ${k}: error: ${error}")
    }

  }

  def rmse(data: RDD[LabeledPoint], model: LinearRegressionModel): Double = {
    val numberExamples = data.count()

    // 数据标准化API调用
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(_.features))
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))

    // 对样本进行预测
    val labelAndPredict = scaledData.map { lp => (lp.label, model.predict(lp.features)) }
    // 计算误差
    val loss = labelAndPredict.map {
      case (l, p) =>
        val err = p - l
        err * err
    }.reduce(_ + _)
    math.sqrt(loss / numberExamples)
  }

}
