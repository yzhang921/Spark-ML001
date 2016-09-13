/**
 * Created by zyong on 2015/12/9.
 */
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter
import net.liftweb.json._
import net.liftweb.json.Serialization.{read, write}

/**
 * Requires ScalaTest and JUnit4.
 */
@RunWith(classOf[JUnitRunner])
class ScalaTest extends FunSuite with BeforeAndAfter {
  test("splitCamelCase works on FooBarBaz") {
      println("Hello World")
  }
}
