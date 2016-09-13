package matrix02
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by zyong on 2016/6/22.
 */
object APITest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val m1 = DenseMatrix.zeros[Double](2, 3)

  }
}
