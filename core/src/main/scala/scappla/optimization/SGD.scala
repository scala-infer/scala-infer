package scappla.optimization

import scappla.Real

class SGD(val debug: Boolean = false, lr: Double) extends Optimizer {

  override def param(initial: Double, name: Option[String]): Real = {
    new Real {

      private var iter: Int = 0

      private var value: Double = initial

      override def v: Double = value

      override def dv(dv: Double): Unit = {
        iter += 1
        value = value + dv * lr / iter
        if (debug) {
          println(s"    SGD (${name.getOrElse("")}) $iter: $value ($dv)")
          //          new Exception().printStackTrace()
        }
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
