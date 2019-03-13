package scappla.distributions

import scappla.Real._
import scappla.Functions.{log, sum}
import scappla.tensor.{DataOps, Dim, GreaterThan, Tensor}
import scappla.{Expr, Score}

import scala.util.Random

case class Categorical[S <: Dim[_], D: DataOps](p: Expr[Tensor[S, D]]) extends Distribution[Int] {

  import scappla.Real._
  import scappla.tensor.TensorExpr._

  private val total = sum(p)

  override def sample(): Int = {
    val draw = Random.nextDouble() * total.v
    val cs = cumsum(p, p.v.shape) - broadcast(draw.const, p.v.shape)
    count(cs, GreaterThan(0f))
  }

  override def observe(value: Int): Score = {
    import scappla.InferField._
    val vals = collect(p.v)
    log(vals(value) / total)
  }

}

