package scappla.optimization
import org.scalatest.FlatSpec

import scala.math.sqrt
import scala.util.Random
import scappla.Real

class OptimizerTest extends FlatSpec {

  class MyOptimizer() extends Optimizer {
    override def param(initial: Double, lr: Double, name: Option[String]): Real = {
      new Real {

        private var iter: Int = 1

        private var v_avg = 0.0
        private var dv_avg = 0.0
        private var vdv_avg = 0.0
        private var vv_avg = 0.0
        private var dvdv_avg = 0.0

        private var value: Double = initial

        override def v: Double = value

        override def dv(dv: Double): Unit = {
          if (iter == 1) {
            v_avg = value
            vv_avg = value * value
            dv_avg = dv
            vdv_avg = value * dv

            value = value + dv * lr
          } else {
            val decay = 1.0 / scala.math.pow(iter + 0.0, 1.0)
//            val decay = 0.01
//            println(s"decay: $decay")
            def newAvg(avg: Double, update: Double): Double = {
              decay * update + (1.0 - decay) * avg
            }
            v_avg = newAvg(v_avg, value)
            vv_avg = newAvg(vv_avg, value * value)
            dv_avg = newAvg(dv_avg, dv)
            vdv_avg = newAvg(vdv_avg, value * dv)

            val v_tau = (vdv_avg - v_avg * dv_avg) / (vv_avg - v_avg * v_avg)
            val max = v_avg - dv_avg / v_tau

            val delta_v = dv - (value - max) / v_tau
            dvdv_avg = newAvg(dvdv_avg, delta_v * delta_v)

            val sigma = scala.math.sqrt(math.abs(dvdv_avg / v_tau))
//            value = max - (value - max) / 2
            value = max + Random.nextGaussian() * sigma / iter
          }

//          println(s"v_avg: ${v_avg}")
//          println(s"vv_avg: ${vv_avg}")
//          println(s"dv_avg: ${dv_avg}")
//          println(s"vdv_avg: ${vdv_avg}")
          iter += 1
        }

        override def toString: String = s"Param@${hashCode()}"
      }

    }
  }


  it should "converge quickly" in {
    val optimizer = new Adam()
//    val optimizer = new MyOptimizer()
//    val optimizer = new SGD()
//    val optimizer = new SGDMomentum()
    val param = optimizer.param(0.0, 1.0)
    for { i <- Range(0, 200)} {
      val value = param.v
      val dv = -(value - 0.2) * 1.2 + 10.0 * Random.nextGaussian()
      param.dv(dv)
      println(s"$value $dv")
    }
  }
}
