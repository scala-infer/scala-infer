package scappla.optimization

import scappla.{BaseField, ValueField, Value}

class Adam(alpha: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) extends Optimizer {

  override def param[X, S](
    initial: X,
    name: Option[String]
  )(implicit
    base: BaseField[X, S],
    expr: ValueField[X, S]
  ): Value[X] = {
    new Value[X] {

      private var iter: Int = 1

      private val shape = base.shapeOf(initial)
      private var value = initial
      private var g_avg = base.fromDouble(0, shape)
      private var gg_avg = base.fromDouble(0, shape)

      private val alphaS = base.fromDouble(alpha, shape)
      private val beta1S = base.fromDouble(beta1, shape)
      private val beta2S = base.fromDouble(beta2, shape)
      private val epsilonS = base.fromDouble(epsilon, shape)
      private val oneMinBeta1S = base.fromDouble(1 - beta1, shape)
      private val oneMinusBeta2s = base.fromDouble(1 - beta2, shape)

      override def v: X = value

      override def dv(dv: X): Unit = {
        import base._

        g_avg = beta1S * g_avg + oneMinBeta1S * dv
        gg_avg = beta2S * gg_avg + oneMinusBeta2s * dv * dv

        val g_hat = g_avg / base.fromDouble(1.0 - math.pow(beta1, iter), shape)
        val gg_hat = gg_avg / base.fromDouble(1.0 - math.pow(beta2, iter), shape)

        value = value + alphaS * g_hat / (base.sqrt(gg_hat) + epsilonS)
        iter = iter + 1
      }

      override def buffer = expr.buffer(this)
    }
  }
}
