package scappla.optimization

import scappla.{BaseField, ValueField, Value}

trait Optimizer {

  def param[X, S](initial: X, name: Option[String] = None)(implicit base: BaseField[X, S], expr: ValueField[X, S]): Value[X]
}
