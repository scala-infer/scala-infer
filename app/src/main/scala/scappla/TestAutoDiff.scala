package scappla

import Functions._

object SUT {
  val square = autodiff { (z: Double) => z * z }
  val fn = autodiff { (z: Double) => z + z * square(z) }
}

object TestAutoDiff extends App {

    val sgd = new SGD()
    val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
    val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val sprinkle = infer[Boolean, Boolean] {
      (rain: Boolean) =>
        if (rain) {
          sample(Bernoulli(0.01), inRain)
        } else {
          sample(Bernoulli(0.4), noRain)
        }
    }

    val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val model = infer[Boolean] {

      val mrain = sample(Bernoulli(0.2), rainPost)
      val sprinkled = sample(sprinkle(mrain))

      val p_wet = (mrain, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true,  false) => 0.8
        case (false, false) => 0.0
      }

      // bind model to data / add observation
      observe(Bernoulli(p_wet), true)

      // return quantity we're interested in
      mrain
    }

    val n_rain = Range(0, 10000).map { _ =>
      sample(model)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")

  /*
  val bw = DValue.ad {
    (z: Double) => {
      val x = z + z * z * z
      x
    }
  }

  val z = new DVariable(2.0)
  val bwz = bw(z)
  bwz.dv(1.0)
  bwz.complete()
  assert(bwz.v == 10.0)
  assert(z.grad == 13.0)
  */

  /*
  val z = new DVariable(2.0)
  val bwz = SUT.fn(z)
  bwz.dv(1.0)
  bwz.complete()
  assert(bwz.v == 10.0)
  assert(z.grad == 13.0)
  */

  /*
  val bla = new Function1[Double, Double] {
    def apply(x: Double) = x * x
  }
  */

/*
  val fn = autodiff {
    (z: Double) =>
      z + pow(2.0, z)
  }
*/

}
