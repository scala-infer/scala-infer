package scappla.optimization

import scappla.{BaseField, Value}

class SGDMomentum(val mass: Int = 10, lr: Double = 1.0) extends Optimizer {

  override def param[@specialized(Float, Double) X, S](initial: X, shp: S, name: Option[String])(implicit ev: BaseField[X, S]): Value[X, S] = {
    new Value[X, S] {

      private var iter: Int = 0

      private var value: X = initial
      private var momentum: X = ev.fromInt(0, shp)

      private val massS = ev.fromDouble(mass, shp)
      private val lrS = ev.fromDouble(lr, shp)

      override val field = ev

      override val shape = shp

      override def v: X = value

      override def dv(dv: X): Unit = {
        iter += 1
        momentum = field.plus(
          momentum,
          field.div(
            field.minus(dv, momentum),
            massS
          )
        )
        value = field.plus(
          value,
          field.times(momentum,
            field.div(lrS, ev.fromInt(iter, shape))
          )
        )
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
