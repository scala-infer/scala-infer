package scappla.optimization

import scappla.Real

trait Optimizer {

  def param(initial: Double, name: Option[String] = None): Real
}
