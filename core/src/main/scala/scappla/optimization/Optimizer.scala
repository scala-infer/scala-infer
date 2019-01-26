package scappla.optimization

import scappla.{BaseField, Expr}

trait Optimizer {

  def param[X, S](initial: X, name: Option[String] = None)(implicit lf: BaseField[X, S]): Expr[X]
}
