package class02

import breeze.linalg.DenseVector
import breeze.numerics._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{MLUtils, LinearDataGenerator}
import org.apache.spark.{SparkContext, SparkConf}
import breeze.linalg.DenseMatrix
import breeze.linalg.{*, DenseVector, DenseMatrix}
import breeze.numerics._
import breeze.stats._
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by zyong on 2016/6/23.
 */
/*object RMSE2 {

  def main(args: Array[String]) {

  }
}*/


object RMSE2 {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Week02Ex02").setMaster("local[2]")
    val sc = new SparkContext(conf)
    // 产生随机 RDD
    var nums = 100;
    var features = 50;
    //生成数据
    val featuresMatrix = DenseMatrix.rand[Double](nums, features)
    val labelMatrix = DenseMatrix.rand[Double](nums, 1)
    //求均值和方差
    val featuresMean = mean(featuresMatrix(::, *)).toDenseVector
    val featuresStddev = stddev(featuresMatrix(::, *)).toDenseVector
    //归一化
    //归一化
    featuresMatrix(*, ::) -= featuresMean
    featuresMatrix(*, ::) /= featuresStddev
    //增加截距
    val intercept = DenseMatrix.ones[Double](featuresMatrix.rows, 1)
    val train = DenseMatrix.horzcat(intercept, featuresMatrix)
    //计算结果
    val w = DenseMatrix.rand[Double](features + 1, 1)
    val A = (train * w).asInstanceOf[DenseMatrix[Double]]
    val probability = 1.0 / (exp(A * -1.0) + 1.0)
    //计算 RMSE
    val RMSE = sqrt(mean(pow(probability - labelMatrix, 2)))
    println("测试 RMSE => " + RMSE)
  }

}
