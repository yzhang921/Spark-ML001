package com.ctrip.fin.bo

/**
 * Created by zyong on 2015/12/9.
 */

import com.ctrip.fin.bi.bo.UserProfile
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, write}

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite}

import scala.collection.mutable

/**
 * Requires ScalaTest and JUnit4.
 */
@RunWith(classOf[JUnitRunner])
class LiftJsonTest extends FunSuite with BeforeAndAfter {

  case class Order(orderNo: String, createDate: String, paymentWay: String, isSubmit: Int)
  case class User(name: String, age: Int, orderArray: Map[String, Order], payways: Set[String])

  class P(val name: String, var age: Int)

  val p = new P("a", 1)

  test("First Serialization/DeSerial Test") {
    implicit val formats = Serialization.formats(NoTypeHints)

    val user1 = User("Mary", 5,
      Map(
        "111" -> Order("111", "2015-12-09", "ccard", 0),
        "222" -> Order("222", "2015-12-09", "alipay", 0)
      ),
      Set("alipay", "ccard")
    )
    val u1jsonStr: String = write(user1)

    println(u1jsonStr)

    val u = read[User](u1jsonStr)
    println(u)

  }

}
