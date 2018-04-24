package scappla

import org.scalatest.FlatSpec

class DValueSpec extends FlatSpec {

  import Functions._

  "The ad macro" should "compute the backward gradient of a polynomial" in {
    val fn = autodiff { (z: Double) => z + z * z * z }

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 10.0)

    value.dv(1.0)
    value.complete()

    assert(variable.grad == 13.0)
  }

  it should "compute the backward gradient of the log" in {
    val fn = autodiff { (z: Double) => z * log(z) }

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 2.0 * scala.math.log(2.0))

    value.dv(1.0)
    value.complete()

    assert(variable.grad == scala.math.log(2.0) + 1.0)

  }

  it should "compute gradient of pow for base" in {
    val fn = autodiff { (z: Double) => z + pow(z, 3.0) }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)
    value.complete()

    val exact: Double => Double = z => 1.0 + 3 * pow(z, 2.0)
    assert(variable.grad == exact(0.5))
  }

  it should "compute gradient of Math.pow for exponent" in {
    val fn = autodiff { (z: Double) =>
      z + pow(2.0, z)
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)
    value.complete()

    val exact: Double => Double =
      z => 1.0 + log(2.0) * pow(2.0, z)
    assert(variable.grad == exact(0.5))
  }

  it should "compute gradient with a block in body" in {
    val fn = autodiff {
      (z: Double) => {
        val x = z * z * z
        x
      }
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)
    value.complete()

    assert(variable.grad == 0.75)
  }

  it should "compose gradients" in {
    val square = autodiff {
      (x: Double) => x * x
    }
    val plus_x = autodiff {
      (x: Double) => x + square(x)
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = plus_x(variable)
    value.dv(1.0)
    value.complete()

    assert(variable.grad == 2.0)
  }

}
