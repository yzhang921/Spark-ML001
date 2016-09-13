package com.ctrip.fin.bo

import com.ctrip.fin.bi.bo._
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.{read, write}
import org.json4s.jackson.JsonMethods._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite}


/**
 * Created by zyong on 2015/12/9.
 */
@RunWith(classOf[JUnitRunner])
class UserProfileTest extends FunSuite with BeforeAndAfter {

  implicit val formats = DefaultFormats

  def serProfile(up: UserProfile): String = {
    write(up.getCaseCopy)
  }

  val o1 = new Order(
    orderId = "111",
    uid = "Peter",
    createDate = "2015-12-08",
    orderAmount = 200.0,
    paymentWay = "alipay"
  )

  val up1 = new UserProfile(o1)

  test("updateProfile") {
    println(serProfile(up1))

    assertResult("Peter") { up1.uid }
    assertResult(200.0) { up1.consumeAbility.totalAmount }
    assertResult(1) { up1.consumeAbility.totalOrderCount }
    assertResult(1) { up1.consumeHabit.usedPayment.size }

    val o2 = new Order(
      orderId = "222",
      uid = "Peter",
      createDate = "2015-12-08",
      orderAmount = 150.0
    )
    up1.updateProfile(o2)
    println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>After Update Profile")
    println(serProfile(up1))

    assertResult("Peter") { up1.uid }
    assertResult(350.0) { up1.consumeAbility.totalAmount }
    assertResult(2) { up1.consumeAbility.totalOrderCount }
    assertResult(1) { up1.consumeHabit.usedPayment.size }

  }

  test("Update Dup-Order") {
    val o3 = new Order(
      orderId = "222",
      uid = "Peter",
      createDate = "2015-12-08",
      orderAmount = 150.0
    )

    up1.updateProfile(o3)
    assertResult("Peter") { up1.uid }
    assertResult(350.0) { up1.consumeAbility.totalAmount }
    assertResult(2) { up1.consumeAbility.totalOrderCount }
    assertResult(1) { up1.consumeHabit.usedPayment.size }
  }

  test("mergeUidCombiner") {

    println("=====================[Test mergeUidCombiner]============================")

    val o2 = new Order(
      orderId = "333",
      uid = "Peter",
      createDate = "2015-12-08",
      orderAmount = 400.0,
      paymentWay = "weichat"
    )

    val up2 = new UserProfile(o2)

    println(serProfile(up1))
    up1.mergeUidCombiner(up2)
    println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>After Merge Profile")
    println(serProfile(up1))

    println(">>>>>>>ProfileLite")
    println(write(up1.getCaseCopyLite))
    println(write(up1.historyOrders))
    assertResult(750.0) { up1.consumeAbility.totalAmount }
    assertResult(3) { up1.consumeAbility.totalOrderCount }
    assertResult(2) { up1.consumeHabit.usedPayment.size }
  }

  test("serialization") {

    println("=====================[Test de-serialization]============================")

    val profileLite = """
    {
      "uid":"m139666918",
      "consumeAbility":{
        "totalOrderCount":5,
        "totalAmount":1382.5
      },
      "consumeHabit":{
        "usedPayment":[
        "EB_MobileAlipay"
        ]
      }
    }"""

    val historyOrders ="""{
        "1598587508":{
        "orderId":"1598587508",
        "uid":"m139666918",
        "createDate":"2015-12-02 12:52:44",
        "paymentWay":"EB_MobileAlipay",
        "orderAmount":197.5,
        "isSubmit":0,
        "isApply":0,
        "isApplySuccess":0
      },
        "1609864931":{
        "orderId":"1609864931",
        "uid":"m139666918",
        "createDate":"2015-12-04 07:07:06",
        "isSubmit":0,
        "isApply":0,
        "isApplySuccess":0
      },
        "1599377876":{
        "orderId":"1599377876",
        "uid":"m139666918",
        "createDate":"2015-12-03 12:39:04",
        "paymentWay":"EB_MobileAlipay",
        "orderAmount":395.0,
        "isSubmit":0,
        "isApply":0,
        "isApplySuccess":0
      },
        "1609941811":{
        "orderId":"1609941811",
        "uid":"m139666918",
        "createDate":"2015-12-04 07:04:33",
        "paymentWay":"EB_MobileAlipay",
        "orderAmount":395.0,
        "isSubmit":0,
        "isApply":0,
        "isApplySuccess":0
      },
        "1611672935":{
        "orderId":"1611672935",
        "uid":"m139666918",
        "createDate":"2015-12-05 00:15:55",
        "paymentWay":"EB_MobileAlipay",
        "orderAmount":395.0,
        "isSubmit":0,
        "isApply":0,
        "isApplySuccess":0
      }
      }"""

      val pfl= UserProfile.parseProfLiteJson(profileLite)
      println(write(pfl))
      val ho = UserProfile.parseHistoryOrderJson(historyOrders)
      println(ho.size)
      ho.foreach{case (oid, order) =>
        println(oid, order)
      }

      val pflByDeser = UserProfile("m139666918", profileLite, historyOrders)

      println(pflByDeser.toString)
  }

}
