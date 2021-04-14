package scappla.distributions

import scappla.Functions.log
import scappla.{Expr, Interpreter, Real, Score}

import scala.util.Random

case class Bernoulli(pExpr: Expr[Double, Unit]) extends Distribution[Boolean] {

  override def sample(interpreter: Interpreter): Boolean = {
    val p = interpreter.eval(pExpr)
    Random.nextDouble() < p.v
  }

  override def observe(interpreter: Interpreter, value: Boolean): Score = {
    val p = interpreter.eval(pExpr)
    if (value) log(p).buffer else log(1.0 - p).buffer
  }

}
