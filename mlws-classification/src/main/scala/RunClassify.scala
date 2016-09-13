import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by zyong on 2015/11/13.
 */
object RunClassify {
  def main(args: Array[String]) {

    val sc = new SparkContext(new SparkConf().setAppName("RunClassify").setMaster("local[2]"))

    val rawData = sc.textFile("D:\\Research\\BI@Home\\SparkClass\\MachineLearningWithSpark\\data\\train.tsv")

    val records = rawData.filter(!_.startsWith("\"url\"")).map(_.split("\t"))

    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }

    data.cache()

    val numData = data.count
    println(s"Count of data: ${numData}")

    // note that some of our data contains negative feature vaues. For naive Bayes we convert these to zeros
    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    val numIterations = 10
    val maxTreeDepth = 5

    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
    val svmModel = SVMWithSGD.train(data, numIterations)
    val nbModel = NaiveBayes.train(nbData)
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)


    println("Predication first element: " + lrModel.predict(data.first().features))

    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum

    val lrAccuracy = lrTotalCorrect / data.count

    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum

    val svmAccuracy = svmTotalCorrect / data.count

    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum

    val nbAccuracy = nbTotalCorrect / data.count

    val dtTotalCorrect = nbData.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum

    val dtAccuracy = dtTotalCorrect / data.count

    // 构造预测值和真实值tuple (预测值, 真实值)
    val metrics = Seq(lrModel, svmModel).map { model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    // again, we need to use the special nbData for the naive Bayes metrics
    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = nbData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }


    // here we need to compute for decision tree separately since it does
    // not implement the ClassificationModel interface
    val dtMetrics = Seq(dtModel).map { model =>
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val allMetrics = metrics ++ nbMetrics ++ dtMetrics

    allMetrics.foreach { case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }


    // 数据正交化
    val vectors = data.map(lp => lp.features)
    //val matrix = new RowMatrix(data.map(_.features))
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))


    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    val lrPredictionsVsTrue = scaledData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC

    println( f"""${lrModelScaled.getClass.getSimpleName}
               | Accuracy: ${lrAccuracyScaled * 100}%2.4f%%
               | Area under PR: ${lrMetricsScaled.areaUnderPR * 100.0}%2.4f%%
               | Area under ROC: ${lrMetricsScaled.areaUnderROC * 100.0}%2.4f%%
             """.stripMargin)


    //Additional features 处理分类属性
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
    println(dataCategories.first)

    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(_.features))
    val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))

    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum

    // 内置验证模型
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }

    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)

    println( f"""${lrModelScaledCats.getClass.getSimpleName}
               | Accuracy: ${lrTotalCorrectScaledCats/numData * 100}%2.4f%%
               | Area under PR: ${lrMetricsScaledCats.areaUnderPR * 100.0}%2.4f%%
               | Area under ROC: ${lrMetricsScaledCats.areaUnderROC * 100.0}%2.4f%%
             """.stripMargin)

  }
}
