package scappla.distributions

import scappla.Functions.{log, sum}
import scappla._

import scala.util.Random

trait RandomGaussian[D, S] {

  def gaussian(shape: S): D
}

object RandomGaussian {

  implicit val forDouble: RandomGaussian[Double, DoubleShape] =
    new RandomGaussian[Double, DoubleShape] {
      override def gaussian(shape: DoubleShape): Double = Random.nextGaussian()
    }
}

case class Normal[D, S](
    mu: Expr[D],
    sigma: Expr[D]
)(implicit
    shapeOf: ShapeOf[D, S],
    rng: RandomGaussian[D, S],
    numX: LiftedFractional[D, S],
    logImpl: log.Apply[Expr[D], Expr[D]],
    sumImpl: sum.Apply[Expr[D]]
) extends DDistribution[D] {

  import numX.mkNumericOps

  override def sample(): Buffered[D] = {
    new Expr[D] {
      val shape: S = shapeOf(sigma.v)
      val e: Expr[D] = numX.const(rng.gaussian(shape))

      override val v: D =
        (mu + e * sigma).v

      override def dv(v: D): Unit = {
        val sc = sigma.const
        val r = (mu + e * sigma / numX.fromInt(2, shape)) / (sc * sc)
        r.dv(v)
      }

    }.buffer
  }

  override def observe(x: Expr[D]): Score = {
    val shape = shapeOf(sigma.v)
    val e = (x - mu) / sigma
    sum(-log(sigma) - e * e / numX.fromInt(2, shape))
  }

  override def reparam_score(x: Expr[D]): Score = {
    val shape = shapeOf(sigma.v)
    val e = (x - mu.const) / sigma.const
    sum(-log(sigma).const - e * e / numX.fromInt(2, shape))
  }
}
