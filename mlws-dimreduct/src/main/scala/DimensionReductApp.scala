import java.awt.image.BufferedImage
import java.io.File

import breeze.linalg.{DenseMatrix, csvwrite}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by zyong on 2015/12/29.
 */
object DimensionReductApp {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Test Dimension Reduction").setMaster("local[2]")
    val sc = new SparkContext()

    val basePath = "D:/Research/BI@Home/SparkClass/MachineLearningWithSpark/data/lfw/*"
    // val basePath = "/data20/workspace/sparkml/lfw/*"
    val rdd = sc.wholeTextFiles(basePath)

    val first = rdd.first

    val files = rdd.map { case (fileName, content) =>
      fileName.replace("file:", "")
    }

    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))


    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache

    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))

    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)
    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)

    csvwrite(new File("D:/Research/BI@Home/SparkClass/MachineLearningWithSpark/data/pc.csv"), pcBreeze)
  }


  def loadImageFromFile(path: String): BufferedImage = {
    import java.io.File
    import javax.imageio.ImageIO
    ImageIO.read(new File(path))
  }

  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }

  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }

  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }

}


