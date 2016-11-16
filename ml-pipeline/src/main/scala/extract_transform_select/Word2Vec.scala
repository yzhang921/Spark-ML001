package extract_transform_select

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zyong on 2016/11/16.
  */
object Word2Vec extends BasicContext {

  conf.setAppName("Word2Vec")

  def main(args: Array[String]) {
    import org.apache.spark.ml.feature.Word2Vec
      // Input data: Each row is a bag of words from a sentence or document.
      val documentDF = sqlContext.createDataFrame(Seq(
        "Hi I heard about Spark".split(" "),
        "I wish Java could use case classes".split(" "),
        "Logistic regression models are neat".split(" ")
      ).map(Tuple1.apply)).toDF("text")
      documentDF.show(10)

      // Learn a mapping from words to Vectors.
      val word2Vec = new Word2Vec()
        .setInputCol("text")
        .setOutputCol("result")
        .setVectorSize(3)
        .setMinCount(0)
      val model = word2Vec.fit(documentDF)
      val result = model.transform(documentDF)
      result.show(10)
      result.select("result").take(3).foreach(println)
  }
}
