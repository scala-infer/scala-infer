package scappla.optimization

import scappla.Real

trait Optimizer {

  def param(initial: Double, lr: Double, name: Option[String] = None): Real
}
