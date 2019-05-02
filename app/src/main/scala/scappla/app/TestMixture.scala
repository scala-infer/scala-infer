package scappla.app

import scappla.Functions.{exp, sigmoid}
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, Guide, ReparamGuide}
import scappla.optimization.Adam
import scappla._

import scala.util.Random

object TestMixture extends App {

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
      Param(mu),
      exp(Param(logSigma))
    ))
  }

  // val sgd = new SGDMomentum(mass = 100)
  val pPost = newGlobal(0.0, 0.0)
  val mu1Post = newGlobal(-1.0, 0.0)
  val mu2Post = newGlobal(1.0, 0.0)
  val sigmaPost = newGlobal(0.0, 0.0)

  val intercept = Param(0.0)
  val slope = Param(1.0)

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

  val sgd = new Adam(0.1)
  val interp = new OptimizingInterpreter(sgd)

  // prepare
  Range(0, 1000).foreach { i =>
    interp.reset()
    model.sample(interp)
  }

  // print some samples
  println("SAMPLES:")
  Range(0, 10).foreach { i =>
    interp.reset()
    val (p, mu1, mu2, sigma) = model.sample(interp)
    println(s"${sigmoid(p.v)}, ${mu1.v}, ${mu2.v}, ${sigma.v}")
  }

  println(s"INTERCEPT: ${interp.eval(intercept).v}, SLOPE: ${interp.eval(slope).v}")

  /*
  // print assignments
  println("ASSIGNMENTS")
  dataWithDist.foreach {
    case (x, param, _) =>
      println(s"$x ${sigmoid(param.v)}")
  }
  */

}
