package scappla.guides

import scappla._
import scappla.distributions.Distribution
import scappla.Interpreter
import scappla.Functions._

trait AutoRegressiveGuideCreator {

  def orig: Guide[Real]

  protected def createGuide(params: Seq[RealExpr], values: Seq[Real]) = new Guide[Real] {

    override def sample(interpreter: Interpreter, prior: Distribution[Real]): scappla.Variable[Real] = {
      val Variable(upstream, node) = orig.sample(interpreter, prior)
      val pv = params.zip(values).map { case (p, v) =>
        interpreter.eval(p) * v
      }.reduce { _ + _ }
      Variable(upstream + pv, node)
    }
  }
}

case class AutoRegressive1(orig: Guide[Real], param: RealExpr)
    extends AutoRegressiveGuideCreator {

  def guide(value: Real): Guide[Real] =
    createGuide(Seq(param), Seq(value))
}

case class AutoRegressive2(orig: Guide[Real], param1: RealExpr, param2: RealExpr)
    extends AutoRegressiveGuideCreator {

  def guide(value1: Real, value2: Real): Guide[Real] =
    createGuide(Seq(param1, param2), Seq(value1, value2))
}

object AutoRegressive {

  def apply(orig: Guide[Real], param: RealExpr) =
    AutoRegressive1(orig, param)

  def apply(orig: Guide[Real], param1: RealExpr, param2: RealExpr) =
    AutoRegressive2(orig, param1, param2)
}
