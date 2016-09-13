package com.fin.bi.bo

import java.text.SimpleDateFormat

import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.{read, write}
import org.json4s.jackson.Serialization.write
import scala.collection.mutable


/**
 * Created by zyong on 2015/12/8.
 */
// 单个订单
case class Order(val orderId: String,
                 val uid: String,
                 val createDate: String,
                 var paymentWay: String = null,
                 var orderAmount: Double = 0.0,
                 var isSubmit: Int = 0,
                 var isApply: Int = 0,
                 var isApplySuccess: Int = 0)

// 消费能力
case class ConsumeAbility(var totalOrderCount: Int = 0, // 历史下单数
                          var totalAmount: Double = 0.0 // 历史下单总金额
                           )

// 消费习惯
case class ConsumeHabit(var paymentway: Map[String, Int],
                         var usedPayment: mutable.HashSet[String] = mutable.HashSet[String]()
                         )

case class UserProfileCC(val uid: String,
                         var consumeAbility: ConsumeAbility,
                         var consumeHabit: ConsumeHabit,
                         var historyOrders: mutable.Map[String, Order]
                          )

// 不包含订单详情
case class UserProfileCCLite(val uid: String,
                             var consumeAbility: ConsumeAbility,
                             var consumeHabit: ConsumeHabit
                              )

// 用户画像
class UserProfile(val uid: String) extends Serializable {

  var consumeAbility: ConsumeAbility = _
  var consumeHabit: ConsumeHabit = _
  var historyOrders: mutable.Map[String, Order] = mutable.Map()

  implicit val formats = new DefaultFormats {
    override def dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  }

  def this(o: Order) {
    this(o.uid)
    this.consumeAbility = ConsumeAbility(totalOrderCount = 1,
      totalAmount = Option(o.orderAmount).getOrElse(0.0))

    this.consumeHabit = Option(o.paymentWay) match {
      case Some(payWay) => new ConsumeHabit(mutable.HashSet(payWay))
      case None => new ConsumeHabit()
    }
    historyOrders.put(o.orderId, o)
  }

  def this(uid: String, pfLiteJson: String, histOrdersJson: String) {
    this(uid)
    val pfLite = UserProfile.parseProfLiteJson(pfLiteJson)
    this.consumeAbility = pfLite.consumeAbility
    this.consumeHabit = pfLite.consumeHabit
    val histOrders = UserProfile.parseHistoryOrderJson(histOrdersJson)
    this.historyOrders ++= histOrders
  }
  /**
   * Combiner 方法：新订单入组
   * @param o
   * @return
   */
  def updateProfile(o: Order): UserProfile = {

    historyOrders.get(o.orderId) match {
      case Some(Ordered) => // 重复订单不处理
      case None =>
        historyOrders.put(o.orderId, o)
        consumeAbility.totalOrderCount += 1
        consumeAbility.totalAmount += Option(o.orderAmount).getOrElse(0.0)
      case _ =>

    }

    Option(o.paymentWay) match {
      case Some(content) => consumeHabit.usedPayment.add(content)
      case None => // 订单还未提交不用处理
    }

    this
  }

  /**
   * Combiner 方法： Merge 两个不同的Profile
   * @param other
   * @return
   */
  def mergeUidCombiner(other: UserProfile): UserProfile = {

    // 合并消费能力
    consumeAbility.totalOrderCount += other.consumeAbility.totalOrderCount
    consumeAbility.totalAmount += other.consumeAbility.totalAmount

    // 合并消费习惯
    consumeHabit.usedPayment ++= other.consumeHabit.usedPayment
    historyOrders ++= other.historyOrders

    this
  }

  /* def toJson: String = {

  }*/

  override def toString: String = {
      write(getCaseCopy)
  }

  def getCaseCopy: UserProfileCC = {
    UserProfileCC(this.uid, this.consumeAbility, this.consumeHabit, this.historyOrders)
  }

  def getCaseCopyLite: UserProfileCCLite = {
    UserProfileCCLite(this.uid, this.consumeAbility, this.consumeHabit)
  }

  def getLiteJson: String = {
    write(getCaseCopyLite)
  }

  def getOrderArrayJson: String = {
    write(historyOrders)
  }


}


object UserProfile {

  implicit val formats = new DefaultFormats {
    override def dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  }

  def apply(o: Order) = new UserProfile(o)

  def apply(uid: String, pfLiteJson: String, histOrdersJson: String) = new UserProfile(uid, pfLiteJson, histOrdersJson)

  def parseProfLiteJson(inJson: String): UserProfileCCLite = {
    val pfl= read[UserProfileCCLite](inJson)
    pfl
  }

  def parseHistoryOrderJson(inJson: String) = {
    val ho= read[Map[String, Order]](inJson)
    ho
  }

}

