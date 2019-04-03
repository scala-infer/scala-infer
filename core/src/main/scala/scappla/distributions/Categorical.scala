package scappla.distributions

import scappla.Functions.{log, sum}
import scappla.tensor.{DataOps, Dim, GreaterThan, Tensor, Index}
import scappla.{Expr, Score}

import scala.util.Random

case class Categorical[S <: Dim[_], D: DataOps](p: Expr[Tensor[S, D]]) extends Distribution[Int] {

  import scappla.tensor.TensorExpr._

  private val total = sum(p)

  override def sample(): Int = {
    val draw = Random.nextDouble() * total.v
    val cs = cumsum(p, p.v.shape)
    count(cs, GreaterThan(draw.toFloat))
  }

  override def observe(index: Int): Score = {
    import scappla.InferField._
    val value = at(p.v, Index[S](List(index)))
    log(value / total)
  }

}

