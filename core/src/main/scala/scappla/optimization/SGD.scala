package scappla.optimization

import scappla.{BaseField, ValueField, Value}

class SGD(val debug: Boolean = false, lr: Double) extends Optimizer {

  override def param[X, S](initial: X, name: Option[String])(implicit ev: BaseField[X, S], expr: ValueField[X, S]): Value[X] = {
    new Value[X] {

      private var iter: Int = 0

      private val shape = ev.shapeOf(initial)
      private var value: X = initial

      private val lrS = ev.fromDouble(lr, shape)

      override def v: X = value

      override def dv(dv: X): Unit = {
        import ev._

        iter += 1
        value = value + dv * lrS / ev.fromInt(iter, shape)
        if (debug) {
          println(s"    SGD (${name.getOrElse("")}) $iter: $value ($dv)")
          //          new Exception().printStackTrace()
        }
      }

      override def toString: String = s"Param@${hashCode()}"
    }
  }
}
