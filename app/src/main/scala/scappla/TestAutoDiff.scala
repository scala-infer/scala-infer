package scappla

import Functions._

import scala.util.Random

object SUT {
  val square = autodiff { (z: Double) => z * z }
  val fn = autodiff { (z: Double) => z + z * square(z) }
}

object TestAutoDiff extends App {

  import DValue._

  /*
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
  */

  val data = {
    val alpha = 1.0
    val sigma = 1.0
    val beta = (1.0, 2.5)

    for {_ <- 0 until 100} yield {
      val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
      val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
      (X, Y)
    }
  }

  val sgd = new SGD()
  val aPost  = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
  val b1Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
  val b2Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
  val sPost  = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

  val model = infer {
    val a   = sample(Normal(0.0, 1.0), aPost)
    val b1  = sample(Normal(0.0, 1.0), b1Post)
    val b2  = sample(Normal(0.0, 1.0), b2Post)
    val err = exp(sample(Normal(0.0, 1.0), sPost))

    data.foreach { entry =>
      val ((x1, x2), y) = entry
      observe(Normal(a + b1 * x1 + b2 * x2, err), y: DValue[Double])
    }

    (a, b1, b2, err)
  }

  // warm up
  Range(0, 10).foreach { i =>
    sample(model)
  }

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
