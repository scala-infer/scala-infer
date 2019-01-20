package scappla.optimization

import scappla.Real

class SGDMomentum(val mass: Int = 10, val debug: Boolean = false) extends Optimizer {

  override def param(initial: Double, lr: Double, name: Option[String]): Real = {
    new Real {

      private var iter: Int = 0

      private var value: Double = initial
      private var momentum: Double = 0.0

      override def v: Double = value

      override def dv(dv: Double): Unit = {
        iter += 1
        momentum = ((mass - 1) * momentum + dv) / mass
        val newValue = value + momentum * lr / iter
        if (debug) {
          println(s"    SGD (${name.getOrElse("")}) $iter: $value (dv: $dv, p: $momentum) => $newValue")
          //          new Exception().printStackTrace()
          scala.math.abs(dv) match {
            case adv: Double if adv > 1.0E8 =>
              assert(false)
            case _ =>
          }
          newValue match {
            case v: Double =>
              if (v.isNaN || v.isInfinite) {
                assert(false)
              }
            case _ =>
          }
        }
        value = newValue
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
