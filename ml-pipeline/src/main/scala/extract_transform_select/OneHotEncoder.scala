package extract_transform_select

/**
  * Created by zyong on 2016/11/16.
  */
object OneHotEncoder extends BasicContext {

  def main(args: Array[String]) {

    import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

    val df = sqlContext.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)
    indexed.show()

    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
    val encoded = encoder.transform(indexed)

    encoded//.select("id", "category", "categoryVec")
      .show()
  }
}
