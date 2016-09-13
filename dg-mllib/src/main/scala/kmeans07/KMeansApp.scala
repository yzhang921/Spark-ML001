package kmeans07

import breeze.linalg._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable.ArrayBuffer
import scala.reflect.io.Path
import scala.util.Random

/**
 * Created by zyong on 2016/7/26.
 */
object KMeansApp {

  def main(args: Array[String]) {


    val basePath = "D:/Research/BI@Home/SparkClass/Spark-ML001/dg-mllib/data/"
    val conf = new SparkConf().setAppName("RunKMenas").setMaster("local[1]")

    val sc = new SparkContext(conf)
    val factor = 15

    val arrayBuffer: ArrayBuffer[DenseVector[Double]] = ArrayBuffer[DenseVector[Double]]()
    for(i <- 10 to 100 by 10) {
      for (j <- 1 to 100) {
        arrayBuffer += DenseVector.rand[Double](2) :* factor.toDouble :+ (i.toDouble)
      }
    }
    val rawData = sc.parallelize(arrayBuffer)

    val inputPath = s"${basePath}kmeans-input"
    val outputPath = s"${basePath}kmeans-output"
    Path(inputPath).deleteRecursively()
    Path(outputPath).deleteRecursively()
    rawData.map(v => "%.4f".format(v(0)) + "," + "%.4f".format(v(1))).saveAsTextFile(inputPath)

    val trainRDD = rawData.map(i => Vectors.dense(i.toArray)).cache()
    val initMode = "K-Means||"
    val model = new KMeans().setInitializationMode(KMeans.K_MEANS_PARALLEL)
      .setK(10)
      .setMaxIterations(100)
      .run(trainRDD)

    val result = trainRDD.map(v => "%.4f".format(v(0)) + "," + "%.4f".format(v(1)) + "," + model.predict(v))
    result.saveAsTextFile(outputPath)

    model.clusterCenters.foreach(println _)

  }

}
