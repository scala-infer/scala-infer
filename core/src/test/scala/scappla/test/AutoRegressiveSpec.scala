package scappla.test

import org.scalatest.FlatSpec
import org.scalatest.exceptions.TestFailedException
import scappla.Functions.exp
import scappla._
import scappla.distributions.Normal
import scappla.guides.{AutoRegressive, ReparamGuide}
import scappla.optimization.Adam


class AutoRegressiveSpec extends FlatSpec {

  it should "infer correlation" in {

    val z1M = Param(0.0)
    val z1S = exp(Param(0.0))
    val z1Post = ReparamGuide(Normal(z1M, z1S))

    val z2M = Param(0.0)
    val z2S = exp(Param(0.0))
    val gamma = Param(0.0)
    val z2Post = AutoRegressive(
      ReparamGuide(Normal(z2M, z2S)),
      gamma
    )

    val s = 1.0
    val x = 3.0
    val model = infer {
      val z1 = sample(Normal(0.0, s), z1Post)
      val z2 = sample(Normal(0.0, s), z2Post.guide(z1))
      observe(Normal(z1 + z2, s), x: Real)
      (z1, z2)
    }

    val interpreter = new OptimizingInterpreter(new Adam(0.1, decay = true))
    assert((0 until 5).exists { _ =>
      try {
        for {_ <- 0 until 10000} {
          val (z1, z2) = model.sample(interpreter)
          interpreter.reset()
        }

        val z1Mv = interpreter.eval(z1M).v
        val z1Sv = interpreter.eval(z1S).v
        val z2Mv = interpreter.eval(z2M).v
        val z2Sv = interpreter.eval(z2S).v
        val gammav = interpreter.eval(gamma).v
        val epsilon = 0.02
        // println(s"$z1Mv, $z1Sv, $z2Mv, $z2Sv, $gammav")
        assert(math.abs(z1Mv - x / 3) < epsilon)
        assert(math.abs(z1Sv - math.sqrt(2.0 / 3)) < epsilon)
        assert(math.abs(z2Mv - x / 2) < epsilon)
        assert(math.abs(z2Sv - math.sqrt(0.5)) < epsilon)
        assert(math.abs(gammav - -0.5) < epsilon)
        true
      } catch {
        case e: TestFailedException =>
          false
      }
    })
  }

  it should "correlate backward" in {

    val z1M = Param(0.0)
    val z1S = exp(Param(0.0))
    val z1Post = ReparamGuide(Normal(z1M, z1S))

    val z2M = Param(0.0)
    val z2S = exp(Param(0.0))
    val gamma = Param(0.0)
    val z2Post = AutoRegressive(
      ReparamGuide(Normal(z2M, z2S)),
      gamma
    )

    val s = 1.0
    val x = 3.0
    val model = infer {
      val z1 = sample(Normal(0.0, s), z1Post)
      val z2 = sample(Normal(z1, s), z2Post.guide(z1))
      observe(Normal(z2, s), x: Real)
      (z1, z2)
    }

    val interpreter = new OptimizingInterpreter(new Adam(0.1, decay = true))
    assert((0 until 5).exists { _ =>
      try {
        for {_ <- 0 until 10000} {
          val (z1, z2) = model.sample(interpreter)
          interpreter.reset()
        }

        val z1Mv = interpreter.eval(z1M).v
        val z1Sv = interpreter.eval(z1S).v
        val z2Mv = interpreter.eval(z2M).v
        val z2Sv = interpreter.eval(z2S).v
        val gammav = interpreter.eval(gamma).v
        val epsilon = 0.02
        println(s"$z1Mv, $z1Sv, $z2Mv, $z2Sv, $gammav")
        assert(math.abs(z1Sv - math.sqrt(2 / 3.0) * s) < epsilon)
        assert(math.abs(z2Sv - 1 / math.sqrt(1 / (s * s) + 1 / (s * s))) < epsilon)
        assert(math.abs(gammav - z2Sv * z2Sv / (s * s)) < epsilon)
        assert(math.abs(z2Mv - x * z2Sv * z2Sv / (s * s)) < epsilon)
        assert(math.abs(z1Mv - (2 / 3.0) * z2Mv) < epsilon)
        true
      } catch {
        case e: TestFailedException =>
          false
      }
    })
  }
}
