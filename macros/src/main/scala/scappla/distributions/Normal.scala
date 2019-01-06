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
  private val shape = shapeOf(sigma.v)
  private val two = numX.fromInt(2, shape)

  override def sample(): Buffered[D] = {
    new Expr[D] with Buffered[D] {
      var refCount = 1

      val e: Expr[D] = numX.const(rng.gaussian(shape))

      val r: Buffered[D] = {
        val sc = sigma.const
        ((mu + e * sigma / two) / (sc * sc)).buffer
      }

      override val v: D =
        (mu + e * sigma).v

      override def dv(x: D): Unit = {
        assert(refCount > 0)
        r.dv(x)
      }

      override def buffer: Buffered[D] = {
        refCount += 1
        this
      }

      override def complete(): Unit = {
        assert(refCount > 0)
        refCount -= 1
        if (refCount == 0) {
          r.complete()
        }
      }
    }
  }

  override def observe(x: Expr[D]): Score = {
    val e = (x - mu) / sigma
    sum(-log(sigma) - e * e / two)
  }

  override def reparam_score(x: Expr[D]): Score = {
    val e = (x - mu.const) / sigma.const
    sum(-log(sigma).const - e * e / two)
  }
}
