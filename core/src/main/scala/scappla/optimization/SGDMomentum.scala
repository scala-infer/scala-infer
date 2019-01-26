package scappla.optimization

import scappla.{BaseField, Expr}

class SGDMomentum(val mass: Int = 10, lr: Double) extends Optimizer {

  override def param[X, S](initial: X, name: Option[String])(implicit ev: BaseField[X, S]): Expr[X] = {
    new Expr[X] {

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
