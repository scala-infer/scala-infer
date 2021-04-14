package scappla.distributions

import scappla.Functions.{log, lgamma}
import scappla.{Expr, Interpreter, Real, Score}

import breeze.stats.distributions.{Poisson => BP}

case class Poisson(lambdaExpr: Expr[Double, Unit]) extends Distribution[Int] {

  override def sample(interpreter: Interpreter): Int = {
    val lambda = interpreter.eval(lambdaExpr)
    BP.distribution(lambda.v).sample()
  }

  override def observe(interpreter: Interpreter, value: Int): Score = {
    val lambda = interpreter.eval(lambdaExpr)
    (value.toDouble * log(lambda) - lambda - lgamma(value + 1.0)).buffer
  }
  
}
