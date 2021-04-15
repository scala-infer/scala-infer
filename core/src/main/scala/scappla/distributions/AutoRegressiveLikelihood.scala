package scappla.distributions

import scappla._
import scappla.distributions.Distribution
import scappla.Interpreter

private class ShiftedLikelihood(
    orig: Likelihood[Real],
    params: Seq[RealExpr],
    values: Seq[Real]
) extends Likelihood[Real] {

  override def observe(interpreter: Interpreter, a: Real): Score = {
    val pv = params
      .zip(values)
      .map {
        case (p, v) =>
          interpreter.eval(p) * v
      }
      .reduce { _ + _ }
    orig.observe(interpreter, a - pv)
  }
}

trait AutoRegressiveLikelihoodCreator {

  def orig: Likelihood[Real]

  protected def createLikelihood(params: Seq[RealExpr], values: Seq[Real]): Likelihood[Real] =
    new ShiftedLikelihood(orig, params, values)
}

case class AutoRegressiveLikelihood1(orig: Likelihood[Real], param: RealExpr)
    extends AutoRegressiveLikelihoodCreator {

  def apply(value: Real): Likelihood[Real] =
    createLikelihood(Seq(param), Seq(value))
}

object AutoRegressiveLikelihood {

  def apply(orig: Likelihood[Real], param: RealExpr) =
    AutoRegressiveLikelihood1(orig, param)
}
