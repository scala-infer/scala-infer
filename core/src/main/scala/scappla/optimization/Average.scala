package scappla.optimization
import scappla.{BaseField, InferField, Expr, Real}

/**
 * Optimizer that uses a decaying learning rate (starting at 1) to compute an optimal value
 */
object Average extends Optimizer {

  override def param[X, S](initial: X, name: Option[String])(implicit ev: BaseField[X, S], expr: InferField[X, S]): Expr[X] = {
    new Expr[X] {

      private val shape = ev.shapeOf(initial)

      private var iter = 0
      private var weight: Double = 0.0
      private var offset: X = ev.fromInt(0, shape)

      private var control = initial

      def v: X = control

      def dv(delta: X): Unit = {
        import ev._

        iter += 1
        val rho = math.pow(iter, -0.5)
        weight = (1.0 - rho) * weight + rho
        offset = ev.fromDouble(1.0 - rho, shape) * offset +
            ev.fromDouble(rho, shape) * (delta + control)
        control = if (weight < 1e-12) {
          ev.fromInt(0, shape)
        }
        else {
          offset / ev.fromDouble(weight, shape)
        }
      }
    }
  }

}
