package scappla.optimization
import scappla.{Expr, Real}

/**
 * Optimizer that uses a decaying learning rate (starting at 1) to compute an optimal value
 */
object Average extends Optimizer {

  override def param(initial: Double, name: Option[String]): Real = {
    new Expr[Double] {

      private var iter = 0
      private var weight: Double = 0.0
      private var offset: Double = 0.0

      private var control = initial

      def v: Double = control

      def dv(delta: Double): Unit = {
        iter += 1
        val rho = math.pow(iter, -0.5)
        weight = (1.0 - rho) * weight + rho
        offset = (1.0 - rho) * offset + rho * (delta + control)
        control = if (weight < 1e-12) {
          0.0
        }
        else {
          offset / weight
        }
      }
    }
  }
}
