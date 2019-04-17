package scappla.optimization

import scappla.{BaseField, InferField, Value}

trait Optimizer {

  def param[X, S](initial: X, name: Option[String] = None)(implicit base: BaseField[X, S], expr: InferField[X, S]): Value[X]
}
