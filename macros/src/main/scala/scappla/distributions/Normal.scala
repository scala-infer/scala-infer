package scappla.distributions

import scappla.Functions.{log, pow, sum}
import scappla._

import scala.util.Random

trait RandomGaussian[D] {

  def gaussian(): D
}

object RandomGaussian {

  implicit val forDouble: RandomGaussian[Double] = new RandomGaussian[Double] {
    override def gaussian(): Double = Random.nextGaussian()
  }
}

case class Normal[D : Fractional : RandomGaussian](
    mu: Expr[D],
    sigma: Expr[D]
)(implicit
    numX: Fractional[Expr[D]],
    logImpl: log.Apply[Expr[D], Expr[D]],
    powImpl: pow.Apply[Expr[D], Expr[D], Expr[D]],
    sumImpl: sum.Apply[Expr[D]]
) extends DDistribution[D] {
  dist =>

  private val numD = implicitly[Fractional[D]]

  override def sample(): Sample[Expr[D]] = new Sample[Expr[D]] {
    import numD.mkNumericOps

    private val x: Expr[D] =
      new Expr[D] {

        private val e: D = implicitly[RandomGaussian[D]].gaussian()

        override val v: D = mu.v + sigma.v * e

        // backprop with inverse fisher, limiting updates by the standard deviation
        override def dv(d: D): Unit = {
          val r : D = d / (sigma.v * sigma.v)
          mu.dv(r)
          sigma.dv(e * r / numD.fromInt(2))
        }
      }

    override val get: Buffered[D] = x.buffer

    override def score: Score = dist.observe(get)

    override def complete(): Unit = get.complete()
  }

  override def observe(x: Expr[D]): Score = {
    import numX.mkNumericOps
    sum(-log(sigma) - pow((x - mu) / sigma, numX.fromInt(2)) / numX.fromInt(2))
  }

  override def reparam_score(x: Expr[D]): Score = {
    import numX.mkNumericOps
    sum(-log(sigma).const - pow((x - mu.const) / sigma.const, numX.fromInt(2)) / numX.fromInt(2))
  }
}
