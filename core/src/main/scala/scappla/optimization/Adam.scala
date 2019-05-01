package scappla.optimization

import scappla.{BaseField, Value}

class Adam(alpha: Double = 0.001, beta1: Double = 0.9, beta2: Double = 0.999, epsilon: Double = 1e-8) extends Optimizer {

  override def param[X, S](
    initial: X,
    shp: S,
    name: Option[String]
  )(implicit
    base: BaseField[X, S]
  ): Value[X, S] = {
    new Value[X, S] {

      private var iter: Int = 1

      private var value = initial
      private var g_avg = base.fromDouble(0, shp)
      private var gg_avg = base.fromDouble(0, shp)

      private val alphaS = base.fromDouble(alpha, shp)
      private val beta1S = base.fromDouble(beta1, shp)
      private val beta2S = base.fromDouble(beta2, shp)
      private val epsilonS = base.fromDouble(epsilon, shp)
      private val oneMinBeta1S = base.fromDouble(1 - beta1, shp)
      private val oneMinusBeta2s = base.fromDouble(1 - beta2, shp)

      override val field = base

      override val shape = shp

      override def v: X = value

      override def dv(dv: X): Unit = {
        import base._

        g_avg = base.plus(
          base.times(beta1S, g_avg),
          base.times(oneMinBeta1S, dv)
        )
        gg_avg = base.plus(
          base.times(beta2S, gg_avg),
          base.times(oneMinusBeta2s, base.times(dv, dv))
        )

        val g_hat = base.div(
          g_avg,
          base.fromDouble(1.0 - math.pow(beta1, iter), shape)
        )
        val gg_hat = base.div(
          gg_avg,
          base.fromDouble(1.0 - math.pow(beta2, iter), shape)
        )

        value = base.plus(
          value,
          base.div(base.times(alphaS, g_hat), (base.plus(base.sqrt(gg_hat), epsilonS)))
        )
        iter = iter + 1
      }
    }
  }
}
