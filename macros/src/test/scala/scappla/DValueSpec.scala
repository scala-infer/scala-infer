package scappla

import org.scalatest.FlatSpec

class DValueSpec extends FlatSpec {

  "The dvalue macro" should "compute the backward gradient of a polynomial" in {
    @dvalue
    def fn(z: Double): Double = z + z * z * z

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 10.0)

    value.dv(1.0)
    value.complete()

    assert(variable.grad == 13.0)
  }

  it should "compute the backward gradient of the log" in {
    @dvalue
    def fn(z: Double): Double = z * DValue.log(z)

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 2.0 * scala.math.log(2.0))

    value.dv(1.0)
    value.complete()

    assert(variable.grad == scala.math.log(2.0) + 1.0)

  }

}
