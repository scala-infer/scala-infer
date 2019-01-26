package scappla.optimization

import scappla.Real

class Adam(alpha: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) extends Optimizer {

  override def param(initial: Double, name: Option[String]): Real = {
    new Real {

      private var iter: Int = 1

      private var value = initial
      private var g_avg = 0.0
      private var gg_avg = 0.0

      override def v: Double = value

      override def dv(dv: Double): Unit = {
        g_avg = beta1 * g_avg + (1.0 - beta1) * dv
        gg_avg = beta2 * gg_avg + (1.0 - beta2) * dv * dv

        val g_hat = g_avg / (1.0 - math.pow(beta1, iter))
        val gg_hat = gg_avg / (1.0 - math.pow(beta2, iter))

        value = value + alpha * g_hat / (math.sqrt(gg_hat) + epsilon)
        iter = iter + 1
      }
    }
  }
}
