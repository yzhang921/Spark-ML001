package extract_transform_select

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SQLContext

/**
  * Created by zyong on 2016/11/16.
  */
object Ets extends BasicContext {

  def main(args: Array[String]) {
    val sentenceData = sqlContext.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.show(10)

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.show(10)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)
  }


}