package matrix02

import breeze.linalg._
import breeze.linalg.DenseVector
import com.github.fommil.netlib.BLAS._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import com.github.fommil.netlib.BLAS.{getInstance => blas}

/**
 * Created by zyong on 2016/6/22.
 */
object RMSE {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val numFeatures = 10
    val numPoints = 20

    val (x, y) = generateVectorData(numFeatures, numPoints)
    println(
      s"""
         |X has ${x.length} vectors as features,
         |every feature has ${x(0).length} points.
       """.stripMargin)

    val dataRDD = sc.parallelize(x).cache()
    dataRDD.count()

    val model = DenseVector.rand[Double](numFeatures + 1)
    val modelRDD = sc.parallelize(model.toArray)
    rmse(predict(dataRDD, modelRDD), y)

    // 矩阵计算
    val featureMatrix = DenseMatrix.zeros[Double](numPoints, numFeatures + 1)
    println(featureMatrix.rows)
    println(featureMatrix.cols)
    var i = 0
    dataRDD.collect().foreach { v =>
      featureMatrix(::, i) := normalize(DenseVector(v))
      i += 1
    }

    rmse(featureMatrix * model, y)
  }

  /**
   * 生成特征向量形式的数据x
   * @param numFeatures 特征向量长度
   * @param numPoints 元素个数
   * @return (x, y)
   */
  def generateVectorData(numFeatures: Int, numPoints: Int): (Array[Array[Double]], DenseVector[Double]) = {

    val random = new Random(40)
    val factorRandom = new Random(50)

    val x = Array.fill[Array[Double]](numFeatures + 1) {
      Array.fill[Double](numPoints)(100 * factorRandom.nextDouble() * random.nextDouble())
    }

    x(0) = Array.fill[Double](numPoints)(1)
    val y = DenseVector(
      Array.fill[Double](numPoints)(random.nextDouble())
    )

    return (x, y)
  }

  /**
   * 向量归一化
   * @param v
   * @return
   */
  def normalize(v: DenseVector[Double]): DenseVector[Double] = {
    if ((max(v)-min(v)) == 0) v else (v - min(v)) / (max(v)-min(v))
  }

  /**
   * 生成预测值向量
   * @param dataRDD X
   * @param modelRDD model
   * @return predicted Y
   */
  def predict(dataRDD: RDD[Array[Double]], modelRDD: RDD[Double]): DenseVector[Double] = {

    val normalized = dataRDD.map { xi =>
      normalize(DenseVector(xi))
    }
    // X*Wt 可以转化为列向量的组合 sum(xi*wi) i: 0->modelRDD.length, reduce之后生成y的预测向量
    val a: DenseVector[Double] = normalized.zip(modelRDD).map { case (xi, wi) =>
      xi * wi
    }.reduce(_ + _)
    a
  }

  /**
   * 均方差
   * @param predict
   * @param actual
   * @return
   */
  def rmse(predict: DenseVector[Double], actual: DenseVector[Double]): Double = {
    val t = predict.toArray.map { p =>
      1.0 / (1 + math.exp(-p))
    }
    (t:*t).sum/t.length
  }


}
