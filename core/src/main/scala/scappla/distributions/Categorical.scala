package scappla.distributions

import scappla.Functions.{log, sum}
import scappla.tensor.{DataOps, Dim, GreaterThan, Index, Tensor}
import scappla.{Expr, Interpreter, Score, Value}

import scala.util.Random

case class Categorical[S <: Dim[_], D: DataOps](pExpr: Expr[Tensor[S, D], S]) extends Distribution[Int] {

  import scappla.tensor.TensorValue._

  private val total = sum(pExpr)

  override def sample(interpreter: Interpreter): Int = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    val draw = Random.nextDouble() * totalv.v
    val cs = cumsum(p, p.v.shape)
    count(cs, GreaterThan(draw.toFloat))
  }

  override def observe(interpreter: Interpreter, index: Int): Score = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    import scappla.ValueField._
    val value = at(p.v, Index[S](List(index)))
    log(value / totalv)
  }

}

