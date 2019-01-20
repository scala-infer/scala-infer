package scappla.optimization

import org.scalatest.FlatSpec

import scala.util.Random

class AdamTest extends FlatSpec {

  it should "converge quickly" in {
    val optimizer = new Adam(alpha = 0.1)
    val param = optimizer.param(0.0)
    for {i <- Range(0, 200)} {
      val value = param.v
      val dv = -(value - 0.2) * 1.2
      param.dv(dv)
    }
    assert(scala.math.abs(param.v - 0.2) < 1e-3)
  }

  it should "converge quickly in the face of noise" in {
    val optimizer = new Adam(alpha = 0.1)
    val param = optimizer.param(0.0)
    for {i <- Range(0, 2000)} {
      val value = param.v
      val dv = -(value - 0.2) * 1.2 + 0.5 * Random.nextGaussian()
      param.dv(dv)
    }
    assert(scala.math.abs(param.v - 0.2) < 1e-2)
  }

}
