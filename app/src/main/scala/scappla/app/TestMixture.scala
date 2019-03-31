package scappla.app

import scappla.Functions.{exp, sigmoid}
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, Guide, ReparamGuide}
import scappla.optimization.Adam
import scappla.{Real, infer, observe, sample}

import scala.util.Random

object TestMixture extends App {

  import Real._

  val data = {
    val p = 0.75
    val mu1 = -0.5
    val mu2 = 1.2
    val sigma = 0.25

    for {_ <- 0 until 500} yield {
      if (Random.nextDouble() < p) {
        Random.nextGaussian() * sigma + mu1
      } else {
        Random.nextGaussian() * sigma + mu2
      }
    }
  }

  /*
  println("DATA:")
  data.foreach(value =>
      println(value)
  )
  */

  def newGlobal(mu: Double, logSigma: Double): Guide[Real] = {
    ReparamGuide(Normal(
      sgd.param(mu),
      exp(sgd.param(logSigma))
    ))
  }

  // val sgd = new SGDMomentum(mass = 100)
  val sgd = new Adam(0.1)
  val pPost = newGlobal(0.0, 0.0)
  val mu1Post = newGlobal(-1.0, 0.0)
  val mu2Post = newGlobal(1.0, 0.0)
  val sigmaPost = newGlobal(0.0, 0.0)

  val intercept = sgd.param(0.0).buffer
  intercept.complete()
  val slope = sgd.param(1.0).buffer
  slope.complete()

  import scappla.InferField._

  val dataWithDist = data.map { datum =>
    val local = intercept + slope * datum
    (datum, BBVIGuide(Bernoulli(sigmoid(local))))
  }
  val model = infer {
    val p = sigmoid(sample(Normal(0.0, 1.0), pPost))
    val mu1 = sample(Normal(0.0, 1.0), mu1Post)
    val mu2 = sample(Normal(0.0, 1.0), mu2Post)
    val sigma = exp(sample(Normal(0.0, 1.0), sigmaPost))

    dataWithDist.foreach[Unit] {
      case (value, guide) =>
        if (sample(Bernoulli(p), guide)) {
          observe(Normal(mu1, sigma), value: Real)
        } else {
          observe(Normal(mu2, sigma), value: Real)
        }
    }

    (p, mu1, mu2, sigma)
  }

  // prepare
  Range(0, 1000).foreach { i =>
    val h_i = intercept.buffer
    val h_s = slope.buffer
    model.sample()
    h_i.complete()
    h_s.complete()
  }

  // print some samples
  println("SAMPLES:")
  Range(0, 10).foreach { i =>
    val (p, mu1, mu2, sigma) = model.sample()
    println(s"${sigmoid(p.v)}, ${mu1.v}, ${mu2.v}, ${sigma.v}")
  }

  println(s"INTERCEPT: ${intercept.v}, SLOPE: ${slope.v}")

  /*
  // print assignments
  println("ASSIGNMENTS")
  dataWithDist.foreach {
    case (x, param, _) =>
      println(s"$x ${sigmoid(param.v)}")
  }
  */

}
