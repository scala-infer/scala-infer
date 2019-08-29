package scappla.optimization

import org.scalatest.FlatSpec

class L2OptimizerTest extends FlatSpec {

  it should "converge to initial value without additional gradients" in {
    val adam = new Adam(1.0)
    val opt = new L2Optimizer(adam, 0.1)
    val param = opt.param(0.0, ())
    for { _ <- 0 until 100 } {
      // - (v - m) ^ 2 / 2
      // => - (v - m) = m - v
      param.dv(5.0 - param.v)
    }
    assert(math.abs(param.v - 5.0) < 0.5)

    for { _ <- 0 until 100 } {
      param.dv(0.0)
    }
    assert(math.abs(param.v) < 0.1)
  }

}