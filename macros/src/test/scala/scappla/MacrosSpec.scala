package scappla

import org.scalatest.FlatSpec
import scappla.distributions.{Bernoulli, Normal}
import scappla.optimization.{SGD, SGDMomentum}

import scala.util.Random

class MacrosSpec extends FlatSpec {

  import Functions._
  import Real._

  it should "allow a model to be specified" in {

    val sgd = new SGD()
    val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
    val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val model = infer {

      val sprinkle = {
        rain: Boolean =>
          if (rain) {
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
      rain
    }

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
  }

  it should "reparametrize doubles" in {
    val sgd = new SGD()
    val muGuide = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

    val model: Model[Real] = infer {
      val mu = sample(Normal(0.0, 1.0), muGuide)

      observe(Normal(mu, Real(1.0)), Real(2.0))

      mu
    }

    // warm up
    Range(0, 10000).foreach { i =>
      sample(model)
    }

    val N = 10000
    val (total_x, total_xx) = Range(0, N).map { i =>
      sample(model).v
    }.foldLeft((0.0, 0.0)) { case ((sum_x, sum_xx), x) =>
      (sum_x + x, sum_xx + x * x)
    }
    val avg_mu = total_x / N
    val var_mu = total_xx / N - avg_mu * avg_mu
    println(s"Avg mu: $avg_mu (${math.sqrt(var_mu)})")
  }

  it should "allow linear regression to be specified" in {

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
          observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
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
      println(s"  $values")
    }
  }

}
