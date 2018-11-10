package scappla

import scappla.Functions.{exp, sigmoid}
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, Guide, ReparamGuide}
import scappla.optimization.SGDMomentum

import scala.util.Random

object TestMixture extends App {

  import Real._

  val data = {
    val p = 0.75
    val mu1 = -0.5
    val mu2 = 1.2
    val sigma = 0.2

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
      sgd.param(mu, 1.0),
      exp(sgd.param(logSigma, 1.0))
    ))
  }

  val sgd = new SGDMomentum(mass = 100)
  val pPost = newGlobal(0.0, -1.0)
  val mu1Post = newGlobal(-1.0, -1.0)
  val mu2Post = newGlobal(1.0, -1.0)
  val sigmaPost = newGlobal(0.0, -1.0)

  val dataWithDist = data.map { datum =>
    (datum, BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0, 100.0)))))
  }
  val model = infer {
    val p = sigmoid(sample(Normal(0.0, 1.0), pPost))
    val mu1 = sample(Normal(0.0, 1.0), mu1Post)
    val mu2 = sample(Normal(0.0, 1.0), mu2Post)
    val sigma = exp(sample(Normal(0.0, 1.0), sigmaPost))

    dataWithDist.foreach[Unit] {
      entry: (Double, Guide[Boolean]) =>
        if (sample(Bernoulli(p), entry._2)) {
          observe(Normal(mu1, sigma), entry._1: Real)
        } else {
          observe(Normal(mu2, sigma), entry._1: Real)
        }
    }

    (p, mu1, mu2, sigma)
  }

  // prepare
  Range(0, 10000).foreach { i =>
    sample(model)
  }

  // print some samples
  println("SAMPLES:")
  Range(0, 10).foreach { i =>
    val l = sample(model)
    val values = (l._1.v, l._2.v, l._3.v, l._4.v)
    println(s"${sigmoid(values._1)}, ${values._2}, ${values._3}, ${exp(values._4)}")
  }

}
