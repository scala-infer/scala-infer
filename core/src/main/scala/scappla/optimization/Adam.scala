package scappla.optimization

import scappla.{BaseField, Expr}

class Adam(alpha: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) extends Optimizer {

  override def param[X, S](initial: X, name: Option[String])(implicit ev: BaseField[X, S]): Expr[X] = {
    new Expr[X] {

      private var iter: Int = 1

      private val shape = ev.shapeOf(initial)
      private var value = initial
      private var g_avg = ev.fromDouble(0, shape)
      private var gg_avg = ev.fromDouble(0, shape)

      private val alphaS = ev.fromDouble(alpha, shape)
      private val beta1S = ev.fromDouble(beta1, shape)
      private val beta2S = ev.fromDouble(beta2, shape)
      private val epsilonS = ev.fromDouble(epsilon, shape)
      private val oneMinBeta1S = ev.fromDouble(1 - beta1, shape)
      private val oneMinusBeta2s = ev.fromDouble(1 - beta2, shape)

      override def v: X = value

      override def dv(dv: X): Unit = {
        import ev._

        g_avg = beta1S * g_avg + oneMinBeta1S * dv
        gg_avg = beta2S * gg_avg + oneMinusBeta2s * dv * dv

        val g_hat = g_avg / ev.fromDouble(1.0 - math.pow(beta1, iter), shape)
        val gg_hat = gg_avg / ev.fromDouble(1.0 - math.pow(beta2, iter), shape)

        value = value + alphaS * g_hat / (ev.sqrt(gg_hat) + epsilonS)
        iter = iter + 1
      }
    }
  }
}
