package scappla.optimization

import scappla.{BaseField, Value}

trait Optimizer {

  def param[X, S](initial: X, shape: S, name: Option[String] = None)(implicit base: BaseField[X, S]): Value[X, S]
}
