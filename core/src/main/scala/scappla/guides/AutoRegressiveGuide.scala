package scappla.guides

import scappla._
import scappla.distributions.Distribution
import scappla.Interpreter

private class ShiftedLikelihood(orig: Likelihood[Real], shift: Real)
    extends Likelihood[Real] {

  override def observe(interpreter: Interpreter, a: Real): Score = {
    orig.observe(interpreter, a + shift)
  }
}

trait AutoRegressiveGuideCreator {

  def orig: Guide[Real]

  protected def createGuide(params: Seq[RealExpr], values: Seq[Real]) =
    new Guide[Real] {

      override def sample(
          interpreter: Interpreter,
          prior: Likelihood[Real]
      ): scappla.Variable[Real] = {
        val pv = params
          .zip(values)
          .map {
            case (p, v) =>
              interpreter.eval(p) * v
          }
          .reduce { _ + _ }
        val Variable(upstream, node) =
          orig.sample(interpreter, new ShiftedLikelihood(prior, pv))
        Variable(upstream + pv, node)
      }
    }
}

case class AutoRegressiveGuide1(orig: Guide[Real], param: RealExpr)
    extends AutoRegressiveGuideCreator {

  def guide(value: Real): Guide[Real] =
    createGuide(Seq(param), Seq(value))
}

case class AutoRegressiveGuide2(
    orig: Guide[Real],
    param1: RealExpr,
    param2: RealExpr
) extends AutoRegressiveGuideCreator {

  def guide(value1: Real, value2: Real): Guide[Real] =
    createGuide(Seq(param1, param2), Seq(value1, value2))
}

object AutoRegressiveGuide {

  def apply(orig: Guide[Real], param: RealExpr) =
    AutoRegressiveGuide1(orig, param)

  def apply(orig: Guide[Real], param1: RealExpr, param2: RealExpr) =
    AutoRegressiveGuide2(orig, param1, param2)
}
