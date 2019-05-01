package scappla.optimization
import scappla.{BaseField, Value, Real}

/**
 * Optimizer that uses a decaying learning rate (starting at 1) to compute an optimal value
 */
object Average extends Optimizer {

  override def param[X, S](initial: X, shp: S, name: Option[String])(implicit bf: BaseField[X, S]): Value[X, S] = {
    new Value[X, S] {

      private var iter = 0
      private var weight: Double = 0.0
      private var offset: X = field.fromInt(0, shp)

      private var control = initial

      override def field = bf

      override val shape = shp

      def v: X = control

      def dv(delta: X): Unit = {
        iter += 1
        val rho = math.pow(iter, -0.5)
        weight = (1.0 - rho) * weight + rho

        offset = field.plus(
          field.times(field.fromDouble(1.0 - rho, shape), offset),
          field.times(field.fromDouble(rho, shape), field.plus(delta, control))
        )
        control = if (weight < 1e-12) {
          field.fromInt(0, shape)
        }
        else {
          field.div(offset, field.fromDouble(weight, shape))
        }
      }
    }
  }

}
