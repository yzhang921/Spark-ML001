import org.apache.spark.mllib.classification.{NaiveBayes, ClassificationModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater, SimpleUpdater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Gini, Entropy}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by zyong on 2015/11/17.
 */
object TunningModelParam {


  def main(args: Array[String]) {

    val sc = new SparkContext(new SparkConf().setAppName("Intro").setMaster("local[2]"))
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

    val scaledCatsData = scaledDataCats(records)
    scaledCatsData.cache()

    //tunningNumberIterations(scaledCatsData)
    //tunningStepSize(scaledCatsData)
    //tunningRangeOfRegulartion(scaledCatsData)
    //compositeLSTunning(scaledCatsData)
    //compositeDTTuning(data)
    tunningNB(nbData)
  }

  /**
   * 调整迭代次数
   * @param scaledCatsData
   */
  def tunningNumberIterations(scaledCatsData: RDD[LabeledPoint]): Unit = {
    val iterResults = Set(1, 5, 10, 50).map { iterNum =>
      val model = trainLSWithParams(scaledCatsData, 0.0, iterNum, new SimpleUpdater, 1.0)
      createMetrics(s"$iterNum iteration", scaledCatsData, model)
    }
    println
    println("====== RegParam:0.0, numberIterations: x, updater: SimpleUpdater, StepSize: 1.0 ======")
    iterResults.foreach { case (label, auc) =>
      println(f"$label, AUC=${auc * 100}%2.2f%%")
    }
  }

  /**
   * 调整步长
   * @param scaledCatsData
   */
  def tunningStepSize(scaledCatsData: RDD[LabeledPoint]) = {

    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainLSWithParams(scaledCatsData, 0.0, numberIterations = 10, new SimpleUpdater, param)
      createMetrics(s"$param step size", scaledCatsData, model)
    }
    println
    println("====== RegParam:0.0, numberIterations: 10, updater: SimpleUpdater, StepSize: x ======")
    stepResults.foreach { case (label, auc) =>
      println(f"$label, AUC = ${auc * 100}%2.2f%%")
    }
  }


  def tunningRangeOfRegulartion(scaledCatsData: RDD[LabeledPoint]) = {

    val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainLSWithParams(scaledCatsData, param, numberIterations = 10, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", scaledCatsData, model)
    }

    println
    println("====== RegParam:x, numberIterations: 10, updater: SquaredL2Updater, StepSize: 1.0 ======")
    regResults.foreach { case (label, auc) => println(f"$label, AUC = ${auc * 100}%2.2f%%")}

  }

  // composite tunning
  def compositeLSTunning(scaledCatsData: RDD[LabeledPoint]) = {
    println("====================综合参数调试==============================")
    for (regParam <- Seq(0.001, 0.01, 0.1, 1.0, 10.0);
         numberIterations <- Seq(1, 5, 10, 50);
         updater <- Seq(new SimpleUpdater(), new SquaredL2Updater(), new L1Updater());
         stepSize <- Seq(0.001, 0.01, 0.1, 1.0, 10.0)
    ) {
      val lr = new LogisticRegressionWithSGD()
      lr.optimizer.setNumIterations(numberIterations)
        .setUpdater(updater)
        .setRegParam(regParam)
        .setStepSize(stepSize)
      val model = lr.run(scaledCatsData)

      val (label, auc) = createMetrics(
        s"regParm: ${regParam}, numberIterations: ${numberIterations}, " +
          s"updater: ${updater.getClass().getName.split("\\.").last}, stepSize: ${stepSize}",
        scaledCatsData, model)
      println(f"$label, AUC = ${auc * 100}%2.2f%%")
    }
  }

  /**
   * 决策树
   * @param data
   */
  def compositeDTTuning(data: RDD[LabeledPoint]): Unit = {
    for (maxDepth <- Seq(1, 2, 3, 4, 5, 10, 20);
         impurity <- Seq(Entropy, Gini)
    ) {
      val model = DecisionTree.train(data, Algo.Classification, impurity, maxDepth)
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      println(f"${maxDepth} tree depth with ${impurity.getClass.getName.split("\\.").last}, AUC = ${metrics.areaUnderROC() * 100}%2.2f%%")
    }

  }

  /**
   * 训练朴素贝叶斯模型
   * @param input
   * @return
   */
  def tunningNB(input: RDD[LabeledPoint]) = {

    Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = new NaiveBayes().setLambda(param).run(input)

      val scoreAndLabel = input.map{ point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabel)

      println(f"$param lambda, AUC = ${metrics.areaUnderROC()*100}%2.2f%%")
    }
  }

  /**
   * 分类1-of-k Feature + 标准化
   * @param records 原始数据分割
   * @return
   */
  def scaledDataCats(records: RDD[Array[String]]): RDD[LabeledPoint] = {
    //Additional features 处理分类属性
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap // Category Map
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

    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(_.features))
    val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))

    scaledDataCats
  }


  /**
   * 接收传入参数
   * @param input
   * @param regParam
   * @param numberIterations
   * @param updater
   * @param stepSize
   * @return
   */
  def trainLSWithParams(input: RDD[LabeledPoint], regParam: Double, numberIterations: Int, updater: Updater, stepSize: Double) = {
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer.setNumIterations(numberIterations)
      .setUpdater(updater)
      .setRegParam(regParam)
      .setStepSize(stepSize)
    lr.run(input)
  }

  /**
   *
   * @param label 模型选择标签
   * @param data 训练数据集
   * @param model 训练模型
   * @return
   */
  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLables = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLables)
    (label, metrics.areaUnderROC())
  }

}


