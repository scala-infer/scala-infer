package scappla.optimization

import scappla._

import org.scalatest.FlatSpec

class BlockLBFGSTest extends FlatSpec {

  /*
  it should "do one dimension" in {
    def f(x: Real): Real = {
      (x - 3.0) * (x - 3.0)
    }
    val N = 100

    {
      val optimizer = new BlockLBFGS(histSize = 5, learningRate = 1.0)
      val xParam = optimizer.param(2.0, ())
      for {iter <- 0 until N} {
        val xBuf = xParam.buffer
        println(s"x: ${xBuf.v}, (iter $iter)")
        val value = f(xBuf)
        value.dv(1.0)
        xBuf.complete()
        optimizer.step()
      }
    }
  }
  */

  it should "converge" in {
    def f(x: Real, y: Real): Real = {
      -(y - x * x) * (y - x * x) - y * y
    }
    val N = 1000

    {
      val optimizer = new BlockLBFGS(histSize = 5, learningRate = 0.05)
      //val optimizer = new Adam(0.1)
      val xParam = optimizer.param(2.0, ())
      val yParam = optimizer.param(1.0, ())
      for {iter <- 0 until N} {
        val xBuf = xParam.buffer
        val yBuf = yParam.buffer
        // println(s"x: ${xBuf.v}, y: ${yBuf.v} (iter $iter)")
        val value = f(xBuf, yBuf)
        value.dv(1.0)
        xBuf.complete()
        yBuf.complete()
        optimizer.step()
      }
    }

    /*
    println("=======================================")

    {
      val optimizer = new BlockLBFGS(histSize = 5, learningRate = 1.0)
      val xParam = optimizer.param(2.0, ())
      val yParam = optimizer.param(1.0, ())
      for {_ <- 0 until N} {
        val xBuf = xParam.buffer
        val yBuf = yParam.buffer
        println(s"x: ${xBuf.v}, y: ${yBuf.v}")
        val value = (2.0: Real) * f(xBuf, yBuf)
        value.dv(1.0)
        xBuf.complete()
        yBuf.complete()
        optimizer.step()
      }
    }
    */
  }
}