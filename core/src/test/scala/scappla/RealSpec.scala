package scappla

import org.scalatest.FlatSpec

import scappla.Functions._

class RealSpec extends FlatSpec {

  case class Param(name: String, var v: Double = 0.0) extends Real {
    var grad = 0.0

    override def dv(v: Double): Unit = grad = v

    override def toString: String = s"$name"
  }

  import ValueField._

  it should "print AST" in {
    val x = Param("x")
    val mu = Param("mu")
    val sigma = Param("sigma")
    println(
      -log(sigma) - (x - mu) * (x - mu) / (2.0 * sigma * sigma)
    )
  }
}
