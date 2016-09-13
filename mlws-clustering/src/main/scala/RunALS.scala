import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating, ALS}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.jblas.DoubleMatrix

/**
 * Created by zyong on 2015/12/1.
 */
object RunALS {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("RunRegression").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val basePath = "D:/Research/BI@Home/SparkClass/MachineLearningWithSpark/data/ml-100k"
    // val basePath = "/data20/workspace/sparkml/ml-100k"

    val movies = sc.textFile(basePath + "/u.item")
    val genres = sc.textFile(basePath + "/u.genre")
    val genreMap = genres.filter(!_.isEmpty).map(_.split("\\|")).map(array => (array(1), array(0))).collectAsMap()

    val titilesAndGenres = movies.map(_.split("\\|")).map { array =>
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter { case (g, idx) =>
        g == "1"
      }.map { case (g, idx) =>
        genreMap(idx.toString)
      }
      (array(0).toInt, (array(1), genresAssigned))
    }

    val rawData = sc.textFile(basePath + "/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map { case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings.cache()
    val alsModel = ALS.train(ratings, 50, 10, 0.1)

    val movieFactors = alsModel.productFeatures.map { case (id, factor) =>
      (id, Vectors.dense(factor))
    }
    val userFactors = alsModel.userFeatures.map { case (id, factor) =>
      (id, Vectors.dense(factor))
    }
    val userVectors = userFactors.map(_._2)

    val predictedRating = alsModel.predict(789, 123)

    val titles = movies.map(_.split("\\|").take(2)).map(array =>
      (array(0).toInt, array(1))
    ).collectAsMap()

    val moviesForUser = ratings.keyBy(_.user).lookup(789)

    //val topKRecs = moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating))
    val userId = 789
    val K = 10
    val topKRecs = alsModel.recommendProducts(userId, K)

    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)

    val itemId = 156
    val itemFactor = alsModel.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)

    val sims = alsModel.productFeatures.map { case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id, sim)
    }

    val sortedSims = sims.top(10)(Ordering.by[(Int, Double), Double] { case (id, sim) => sim})

    sortedSims.map { case (id, sim) =>
      (titles(id), sim)
    }.mkString("\n")

    evaluation(alsModel, ratings)


    val actualMovies = moviesForUser.map(_.product)
  }

  /**
   * 余弦相似度
   * @param vec1
   * @param vec2
   * @return
   */
  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }


  /**
   * 模型评估
   * @param model
   * @param ratings
   */
  def evaluation(model: MatrixFactorizationModel, ratings: RDD[Rating]) = {
    val usersProducts = ratings.map { case Rating(user, product, rating) => (user, product)}

    val predictions = model.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }

    val ratingsAndPredictions = ratings.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)

    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count()
    println("Mean Squared Error = " + MSE)
  }

  /**
   * 用内建工具评估模型
   * @param model
   * @param ratings
   */
  def evaluationWithBuildInFunc(model: MatrixFactorizationModel, ratings: RDD[Rating]) = {

    val usersProducts = ratings.map { case Rating(user, product, rating) => (user, product)}

    val predictions = model.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }

    val ratingsAndPredictions = ratings.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)

    import org.apache.spark.mllib.evaluation.RegressionMetrics
    val predictedAndTrue = ratingsAndPredictions.map {
      case ((user,product), (predicted, actual)) => (predicted, actual)
    }
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)

    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
  }

  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else {
      score / scala.math.min(actual.size, k).toDouble
    }
  }
}
