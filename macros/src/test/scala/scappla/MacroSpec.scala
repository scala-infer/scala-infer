package scappla

import org.scalatest.FlatSpec
import scappla.Functions.{log, pow}

class MacroSpec extends FlatSpec {

  "The backward macro" should "compute the backward gradient of a polynomial" in {
    // c = fn(a, b)
    // ((a, b), dc) => (da, db)
    val bw: (Double, Double) => Double =
    Macro.backward[Double, Double](z => z + z * z * z)
    assert(bw(1.0, 1.0) == 4.0)
    assert(bw(1.0, 0.5) == 1.75)
  }

  it should "compute gradient of pow for base" in {
    val bw: (Double, Double) => Double =
      Macro.backward[Double, Double](z => z + pow(z, 3.0))
    val exact: Double => Double = z => 1.0 + 3 * pow(z, 2.0)
    assert(bw(1.0, 0.5) == exact(0.5))
    assert(bw(1.0, 1.0) == exact(1.0))
  }

  it should "compute gradient of Math.pow for exponent" in {
    val bw: (Double, Double) => Double =
      Macro.backward[Double, Double](z => z + pow(2.0, z))
    val exact: Double => Double = z => 1.0 + log(2.0) * pow(2.0, z)
    assert(bw(1.0, 0.5) == exact(0.5))
    assert(bw(1.0, 1.0) == exact(1.0))
  }

  it should "compute gradient with a block in body" in {
    val bw: (Double, Double) => Double =
      Macro.backward[Double, Double] { z =>
        val x = z * z
        x
      }
    assert(bw(1.0, 1.0) == 2.0)
    assert(bw(1.0, 0.5) == 1.0)
  }

}
