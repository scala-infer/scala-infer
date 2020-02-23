package scappla.distributions

import scappla.Functions.{log, sum}
import scappla.tensor._
import scappla.tensor.Tensor._
import scappla.{Expr, Interpreter, Score, Value}

import scala.util.Random

case class Categorical[S <: Dim[_], D: TensorData](pExpr: Expr[D, S]) extends Distribution[Int] {

  private val total = sum(pExpr)

  override def sample(interpreter: Interpreter): Int = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    val draw = Random.nextDouble() * totalv.v
    val cs = cumsum(p, p.shape)
    val index = p.shape.size - count(cs, GreaterThan(draw.toFloat))
    if (index == p.shape.size) {
      // rounding error - should occur only very rarely
      p.shape.size - 1
    } else {
      index
    }
  }

  override def observe(interpreter: Interpreter, index: Int): Score = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    val value = at(p, Index[S](List(index)))
    log(value / totalv)
  }

}

