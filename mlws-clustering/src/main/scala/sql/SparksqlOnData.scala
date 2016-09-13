package sql

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


/**
 * Created by zyong on 2015/12/3.
 */
object SparksqlOnData {

  // 流派
  case class Genre(genre: String, id: Int)

  // 用户
  case class User(user_id: Int,
                  age: Int,
                  gender: String,
                  occupation: String,
                  zip_code: String)

  // 评价
  case class Rating(user: Int, movie: Int, rating: Double)

  val movieSchemaStr = """movie id | movie title | release date | video release date |
                         $IMDb URL | unknown | Action | Adventure | Animation |
                         $Children's | Comedy | Crime | Documentary | Drama | Fantasy |
                         $Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
                         $Thriller | War | Western |""".stripMargin('$').replaceAll("\n", "")

  val movieSchema = StructType(movieSchemaStr.split("\\|").map(_.trim.replaceAll(" ", "_")).
                      map(fieldName => StructField(fieldName, StringType, true)))

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val basePath = "D:/Research/BI@Home/SparkClass/MachineLearningWithSpark/data/ml-100k"
    // val basePath = "/data20/workspace/sparkml/ml-100k"

    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val moviesRowRDD = sc.textFile(basePath + "/u.item").map(_.split("\\|")).map(p => Row.fromSeq(p.toSeq))
    val moviesDF = sqlContext.createDataFrame(moviesRowRDD, movieSchema)

    val genresDF = sc.textFile(basePath + "/u.genre").
                      map(_.split("\\|")).filter(_.size==2).
                      map(p => Genre(p(0),p(1).toInt)).toDF()

    val ratingDF = sc.textFile(basePath + "/u.data").map(_.split("\t")).
                      map(p => Rating(p(0).toInt, p(1).toInt, p(2).toDouble)).toDF

    val userDF = sc.textFile(basePath + "/u.user").
                    map(_.split("\\|")).
                    map(p => User(p(0).toInt, p(1).toInt, p(2), p(3), p(4))).toDF

    userDF.filter("user_id=789").show
    ratingDF.filter("user=789").
      join(moviesDF, $"movie" === $"movie_id").
      selectExpr("user", "movie_id", "movie_title", "rating").show

    // Register the DataFrames as a table.
    userDF.registerTempTable("user")
    ratingDF.registerTempTable("rating")
    genresDF.registerTempTable("genre")
    moviesDF.registerTempTable("movie")

    sqlContext.sql(
      """SELECT a.user, b.movie_id, b.movie_title, a.rating
        |  FROM rating a
        |  JOIN movie b ON a.movie = b.movie_id
        | WHERE a.user=789
      """.stripMargin).show()


    val ratings = ratingDF.map { row =>
      org.apache.spark.mllib.recommendation.Rating(row.getAs[Int]("user"), row.getAs[Int]("movie"), row.getAs[Double]("rating"))
    }
    ratings.cache()
    val alsModel = ALS.train(ratings, 50, 10, 0.1)

    /**
     * RDD transformations and actions can only be invoked by the driver, not inside of other transformations;
     * for example, rdd1.map(x => rdd2.values.count() * x) is invalid because the values transformation and count
     * action cannot be performed inside of the rdd1.map transformation. For more information, see SPARK-5063.
     */
    /*sqlContext.udf.register("recommend", (userid: Int) =>
      alsModel.recommendProductsForUsers(userid, 3).map(_.product).toString) //  不能注册为UDF因为会出发Action
      */
    sqlContext.udf.register("doubid", (id: Int) => id*2)

    sqlContext.sql(
      """SELECT user_id
        |      ,doubid(user_id)
        |      ,recommend(user_id)
        |      --,cluster(user_id)
        |  FROM user
        |--LIMIT 10
      """.stripMargin).show

    val vctors = userDF.map{ row => Vectors.dense(row.getAs[Int]("user_id")) }
    val cluster = KMeans.train(vctors, 2, 20)
    sqlContext.udf.register("cluster", (id: Int) => cluster.predict(Vectors.dense(id)))


    sqlContext.sql(
    """WITH a AS (
      |  SELECT * FROM movie
      |)
      |SELECT movie_title FROM a WHERE movie_id = 567
    """.stripMargin
    ).show
  }


  def testEditor(sqlContext: SQLContext ): Unit = {
    // 临时表可以注册为TempTable
    sqlContext.sql("SELECT * FROM dim_paydb.dimcardtype limit 10").registerTempTable("cardtype")

    sqlContext.sql(
      """SELECT a.*
        |  FROM dim_paydb.dimcardtype a
        |  JOIN cardtype b ON a.cardcode = b.cardcode
      """.stripMargin).show()

    sqlContext.sql(
      """
        |CREATE TABLE tmp_payquerydb.zy_sparkcrt
        |AS
        |SELECT a.*
        |  FROM dim_paydb.dimcardtype a
        |  JOIN cardtype b ON a.cardcode = b.cardcode
      """.stripMargin)
  }
}
