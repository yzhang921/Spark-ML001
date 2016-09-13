package com.fin.bi.job

import java.text.SimpleDateFormat

import com.fin.bi.bo.{Order, UserProfile}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.DefaultFormats

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization._

/**
 * Created by zyong on 2015/12/8.
 */
object FinUserProfile {

  case class ProfileRow(uid: String, profileMap: String, orderMap: String)

  implicit val formats = new DefaultFormats {
    override def dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("BIFin-UserProfile")
    val sc = new SparkContext(conf)
    val sqlContext = new HiveContext(sc)

    val paymentOrder = sqlContext.sql(
      """
        |SELECT a.uid, orderid, paymentwayidg, creationdate, issubmitg, applypays, amount,
        |       formattedpaytype, formattedsubpaytype
        |  FROM tmp_payquerydb.zy_paymentordersample a
        | WHERE a.inblacklist =0
        | """.stripMargin
    ).map { r =>
      Order(
        orderId = r.getAs[String]("orderid"),
        uid = r.getAs[String]("uid"),
        createDate = r.getAs[String]("creationdate"),
        paymentWay = r.getAs[String]("paymentwayidg"),
        orderAmount = r.getAs[Double]("amount")
      )
    }

    val uidGroup = paymentOrder.map {
      o: Order => (o.uid, o)
    }.combineByKey(
        createUidCombiner,
        mergeUid,
        mergeUidCombiner
      )

    // 把数据注册为临时表
    import sqlContext.implicits._
    uidGroup.map { case (uid, profile) =>
      ProfileRow(uid, profile.getLiteJson, profile.getOrderArrayJson)
    }.toDF().registerTempTable("tmp_pay_userprofile")

    // 保存数据
    sqlContext.sql("""DROP TABLE IF EXISTS tmp_paydb.tmp_spark_userprofile""")
    sqlContext.sql(
      """CREATE TABLE tmp_paydb.tmp_spark_userprofile
        |AS
        |SElECT * FROM tmp_pay_userprofile
      """.stripMargin)

  }

  def createUidCombiner(o: Order): UserProfile = {
    UserProfile(o)
  }

  def mergeUid(profile: UserProfile, o: Order): UserProfile = {
    profile.updateProfile(o)
  }

  def mergeUidCombiner(p1: UserProfile, p2: UserProfile): UserProfile = {
    p1.mergeUidCombiner(p2)
  }

}
