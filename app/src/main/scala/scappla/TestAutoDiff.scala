package scappla

import Functions._
import scappla.distributions.Bernoulli
import scappla.optimization.SGD

import scala.util.Random

/*
object SUT {
  val square = autodiff { (z: Double) => z * z }
  val fn = autodiff { (z: Double) => z + z * square(z) }
}
*/

object TestAutoDiff extends App {

  import Real._

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

  {
    val sgd = new SGD()
    val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
    val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val model = infer {

      val sprinkle = (arg: Boolean) => {
          if (true) {
            sample(Bernoulli(0.01), inRain)
          } else {
            sample(Bernoulli(0.4), noRain)
          }
      }

      val rain = sample(Bernoulli(0.2), rainPost)
      val sprinkled = sprinkle(rain)

      val p_wet = (rain, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true, false) => 0.8
        case (false, false) => 0.001
      }

      // bind model to data / add observation
      observe(Bernoulli(p_wet), true)

      // return quantity we're interested in
      true
    }

    /*

    val N = 10000
    // burn in
    for {_ <- 0 to N} {
      sample(model)
    }

    // measure
    val n_rain = Range(0, N).map { _ =>
      sample(model)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %
    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
    */
  }

/*
  {
    import DValue._

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

    val sgd = new SGDMomentum(mass = 100)
    val aPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val b1Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val b2Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val sPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

    val model = infer {
      val a = sample(Normal(0.0, 1.0), aPost)
      val b1 = sample(Normal(0.0, 1.0), b1Post)
      val b2 = sample(Normal(0.0, 1.0), b2Post)
      val err = exp(sample(Normal(0.0, 1.0), sPost))

      val cb = {
        entry: ((Double, Double), Double) =>
          val ((x1, x2), y) = entry
          observe(Normal(a + b1 * x1 + b2 * x2, err), y: DValue[Double])
      }
      data.foreach[Unit](cb)

      (a, b1, b2, err)
    }

    // warm up
    Range(0, 10000).foreach { i =>
      sample(model)
    }

    // print some samples
    Range(0, 10).foreach { i =>
      val l = sample(model)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  ${values}")
    }
  }
*/

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
