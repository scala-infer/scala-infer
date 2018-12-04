package scappla.distributions

import scappla.Functions.{log, pow}
import scappla._

import scala.util.Random


case class Normal(mu: Real, sigma: Real) extends DDistribution {
  dist =>

  override def sample(): Sample[Real] = new Sample[Real] {

    private val x: Real =
      new Real {
        private val e = Random.nextGaussian()
        private val scale = 5.0

        override val v: Double = mu.v + sigma.v * e

        // backprop with inverse fisher, limiting updates by the standard deviation
        override def dv(d: Double): Unit = {
          val r = scale * math.tanh(d / (sigma.v * scale)) / sigma.v
          mu.dv(r)
          sigma.dv(e * r / 2)
        }
      }

    override val get: RealBuffer = x.buffer

    override def score: Score = dist.observe(get)

    override def complete(): Unit = get.complete()
  }

  val logSigma = new Real {
    val upstream = log(sigma)

    override def v: Double = ???

    override def dv(v: Double): Unit = ???
  }

  override def observe(x: Real): Score = {
    -log(sigma) - pow((x - mu) / sigma, Real(2.0)) / Real(2.0)
  }

  override def reparam_score(x: Real): Score = {
    -log(sigma).const - pow((x - mu.const) / sigma.const, Real(2.0)) / Real(2.0)
  }
}

/*
object Normal {

  def apply(mu: Double, sigma: Double): Normal =
    Normal(Real(mu), Real(sigma))
}
*/
