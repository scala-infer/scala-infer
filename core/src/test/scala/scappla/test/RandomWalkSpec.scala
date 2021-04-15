package scappla.test

import org.scalatest.FlatSpec
import java.io.File
import java.io.PrintWriter

import scala.util.Random
import scappla._
import scappla.Functions._
import scappla.distributions.Normal
import scappla.guides.ReparamGuide
import scappla.optimization.{Adam, BlockLBFGS, SGDMomentum}
import scappla.guides.Guide
import scappla.guides.AutoRegressiveGuide
import scappla.optimization.Optimizer

class RandomWalkSpec extends FlatSpec {

  /**
   * Do a gaussian random walk and verify that the parameters can be inferred.
   * This inference problem can be solved exactly, so that's done here to compare against.
   * (there's a bit of overhead due to the treatment of the last term - the last_s_z correction)
   * 
   * The $y$ variables correspond to the exact solution (whose solution runs the walk backwards).
   * 
   * The prior parameters $s_x$ and $s_z$ are found by Empirical Bayes (evidence maximization),
   * while the hidden steps are found with their standard deviation.  Since successive steps
   * are highly correlated, we use an auto-regressive guide to take care of these correlations
   * (exactly).
   */
  it should "fit volatility" in {
    val rng = new Random(0L)

    val s_z = 0.1
    val s_x = 0.2
    val N = 200

    val hidden = (0 until N).scanLeft(0.0) { case (z, _) =>
      z + rng.nextGaussian() * s_z
    }
    val data = hidden.map { z =>
      (z, z + s_x * rng.nextGaussian())
    }

    val alpha = {
      val b = 1.0 + s_z * s_z / (2 * s_x * s_x)
      b - math.sqrt(b * b - 1.0)
    }

    val s_z_post = math.sqrt(alpha) * s_z
    val last_s_z = s_z_post / math.sqrt(1.0 - alpha * (1.0 + s_z * s_z / (s_x * s_x)))

    val dataWithY = data.reverse.scanLeft(0.0)  {
      case (acc, (_, x)) => alpha * x * s_z * s_z / (s_x * s_x) + alpha * acc
    }.drop(1).reverse.zip(data).map {
      case (y, (z, x)) => (z, x, y)
    }

    val alphaParam = logistic(Param(0.0, "alpha"))
    val dataGuideErr = exp(Param(-1.0, "s_z_post"))
    val dataWithGuides = dataWithY.zipWithIndex.map { case ((z, x, y), i) =>
      val mu = Param(0.0, s"y-$i")
      val guide = ReparamGuide(Normal(mu, dataGuideErr))
      (x, AutoRegressiveGuide(guide, alphaParam), (z, mu, y))
    }

    val volParam = exp(Param(-1.0, "s_z"))
    val errParam = exp(Param(-1.0, "s_x"))
    val model = infer {
      val last_hidden = dataWithGuides.foldLeft(0.0: Real) {
        case (prev, (x, ar, _)) =>
          // prior: (z_i - z_{i-1})^2 / 2 vol^2
          // posterior: (z_i - (\alpha z_{i-1} + mu_i))^2 / 2 \hat vol^2
          val hidden: Real = sample(Normal(prev, volParam), ar.guide(prev))
          // observation: (z_i - x_i)^2 / 2 err^2
          observe(Normal(hidden, errParam), x: Real)
          hidden
      }

      observe(Normal(last_hidden, last_s_z), 0.0: Real)

      last_hidden
    }

     val optimizer = new Adam(0.1, decay = 0.5)
    val interpreter = new OptimizingInterpreter(optimizer)
    for { iter <- 0 until 10000 } {
      val dummy = model.sample(interpreter)
      /*
      if (iter % 100 == 0) {
        val vol = interpreter.eval(volParam)
        val err = interpreter.eval(errParam)
        val dErr = interpreter.eval(dataGuideErr)
        val alpha = interpreter.eval(alphaParam)
        println(s"RANDOMWALK: ${iter}, ${vol.v}, ${err.v}, ${alpha.v}, ${dErr.v}")
      }
      */
      interpreter.reset()
    }
    val dummy = model.sample(interpreter)
    assert(math.abs(interpreter.eval(volParam).v - s_z) < 0.01)
    assert(math.abs(interpreter.eval(errParam).v - s_x) < 0.02)
    assert(math.abs(interpreter.eval(dataGuideErr).v - s_z_post) < 0.01)
    /*
    dataWithGuides.foldLeft((0.0, 0.0)){ case ((sum_mu, sum_y), (x, _, (z, muParam, y))) =>
      val mu = interpreter.eval(muParam)
      val z_mu = alpha * sum_mu + mu.v
      val z_y = alpha * sum_y + y
      // val result = muv
      println(s"HIDDEN: ${z}, $z_mu, ${x}, ${z_y}")
      (z_mu, z_y)
    }
    */

  }
}