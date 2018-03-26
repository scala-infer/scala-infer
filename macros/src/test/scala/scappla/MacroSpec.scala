package scappla

import org.scalatest.FlatSpec
import scappla.Functions.{log, pow}

class MacroSpec extends FlatSpec {

  "The backward macro" should "compute the backward gradient of a polynomial" in {
    // c = fn(a, b)
    // ((a, b), dc) => (da, db)
    @differentiate
    def fn(z: Double): Double = z + z * z * z

    assert(fn(1.0) == 2.0)
    assert(fn.grad(1.0, 1.0) == 4.0)
    assert(fn.grad(0.5, 1.0) == 1.75)
  }

  it should "compute gradient of pow for base" in {
    @differentiate
    def fn(z: Double): Double = z + pow(z, 3.0)

    val exact: Double => Double = z => 1.0 + 3 * pow(z, 2.0)
    assert(fn.grad(1.0, 1.0) == exact(1.0))
    assert(fn.grad(0.5, 1.0) == exact(0.5))
  }

  it should "compute gradient of Math.pow for exponent" in {
    @differentiate
    def fn(z: Double): Double =
      z + pow(2.0, z)

    val exact: Double => Double =
      z => 1.0 + log(2.0) * pow(2.0, z)
    assert(fn.grad(1.0, 1.0) == exact(1.0))
    assert(fn.grad(0.5, 1.0) == exact(0.5))
  }

  it should "compute gradient with a block in body" in {
    @differentiate
    def fn(z: Double): Double = {
      val x = z * z
      x
    }

    assert(fn.grad(1.0, 1.0) == 2.0)
    assert(fn.grad(0.5, 1.0) == 1.0)
  }

}
