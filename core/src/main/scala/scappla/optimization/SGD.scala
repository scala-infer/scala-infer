package scappla.optimization

import scappla.{BaseField, Value}

class SGD(val debug: Boolean = false, lr: Double) extends Optimizer {

  override def param[X, S](initial: X, shp: S, name: Option[String])(implicit ev: BaseField[X, S]): Value[X, S] = {
    new Value[X, S] {

      private var iter: Int = 0

      private var value: X = initial

      override val field = ev

      override val shape = shp

      private val lrS = ev.fromDouble(lr, shape)

      override def v: X = value

      override def dv(dv: X): Unit = {
        iter += 1
        value = ev.plus(
          value,
          field.times(dv, field.div(lrS, ev.fromInt(iter, shape)))
        )
        if (debug) {
          println(s"    SGD (${name.getOrElse("")}) $iter: $value ($dv)")
          //          new Exception().printStackTrace()
        }
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
