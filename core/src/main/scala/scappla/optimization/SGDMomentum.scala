package scappla.optimization

import scappla.{BaseField, ValueField, Value}

class SGDMomentum(val mass: Int = 10, lr: Double = 1.0) extends Optimizer {

  override def param[X, S](initial: X, name: Option[String])(implicit ev: BaseField[X, S], expr: ValueField[X, S]): Value[X] = {
    new Value[X] {

      private var iter: Int = 0

      private val shape = ev.shapeOf(initial)
      private var value: X = initial
      private var momentum: X = ev.fromInt(0, shape)

      private val massS = ev.fromDouble(mass, shape)
      private val lrS = ev.fromDouble(lr, shape)

      override def v: X = value

      override def dv(dv: X): Unit = {
        import ev._

        iter += 1
        momentum = momentum  + (dv - momentum) / massS
        value = value + momentum * lrS / ev.fromInt(iter, shape)
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
