package scappla.optimization

import scappla.{BaseField, Value}

trait Optimizer {

  def param[X, S](
      initial: X,
      shape: S,
      name: Option[String] = None
  )(implicit
      base: BaseField[X, S]
  ): Value[X, S]

  /**
   * Generate parameters for a parameter group.
   * These are optimized together, e.g. by calculating
   * the approximate Hessian.
   */
  def paramGroup(
      initial: List[Double],
      size: Int
  ): List[Value[Double, Unit]] = {
    initial.map {
      param(_, ())
    }
  }

  def step(): Unit = {}
}
