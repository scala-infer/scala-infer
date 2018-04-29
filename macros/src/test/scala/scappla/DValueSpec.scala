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

  /*
  it should "allow a model to be specified" in {

    val sprinkle = infer {
      (rain: Boolean) =>
        if (rain) {
          sample(Bernoulli(0.01).draw(Enumeration))
        } else {
          sample(Bernoulli(0.4).draw(Enumeration))
        }
    }

    val model = infer {

      val rain = sample(Bernoulli(0.2).draw(Enumeration))
      val sprinkled = sample(sprinkle(rain))

      val p_wet = (rain, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true,  false) => 0.8
        case (false, false) => 0.0
      }

      // bind model to data / add observation
      factor(Bernoulli(p_wet).score(true))

      // return quantity we're interested in
      rain
    }

    val n_rain = Range(0, 10000).map { _ =>
      sample(model)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")
  }
  */

  it should "follow the monad pattern" in {
    val px = 0.2
    val py = 0.7
    val success = for {
      x <- Bernoulli(px).draw(Enumeration)
      y <- Bernoulli(py).draw(Enumeration)
    } yield (x, y)

    val N = 10000
    val n_success = (0 to N)
        .map { _ => sample(success) }
        .groupBy(identity)
        .mapValues(_.size)

    def expected(x: Boolean, y: Boolean) = {
      (if (x) px else 1.0 - px) * (if (y) py else 1.0 - py)
    }

    for { ((x, y), count) <- n_success } {
      val n = expected(x, y) * N
      assert(math.abs(count - n) < 3 * math.sqrt(n))
    }
  }

}
