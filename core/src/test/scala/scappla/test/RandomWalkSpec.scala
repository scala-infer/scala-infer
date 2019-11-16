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
import scappla.guides.AutoRegressive
import scappla.optimization.Optimizer

class RandomWalkSpec extends FlatSpec {

  /**
   * For the simple random walk case, we can integrate out the hidden variables exactly.
   * This leaves an exact free energy to optimize - make sure we can do at least that.
   */
  it should "optimize exact free energy" in {
    val hidden = (0 until 500).scanLeft(0.0) { case (v, _) =>
      v + Random.nextGaussian() * 0.1
    }
    val data = hidden.map { v =>
      (v, v + 0.2 * Random.nextGaussian())
    }

    val x = data.map { _._2 }
    val distWeights = (for {
      i <- x.indices
      j <- x.indices
    } yield {
      val dist = if (i < j)
        j - i
      else
        i - j

      (dist, x(i) * x(j))
    }).groupBy(_._1).map {
      case (dist, pairs) =>
        (dist: Real, pairs.map { _._2 }.sum)
    }
    val diag = x.map { xi => xi * xi }.sum

    val one: Real = 1.0
    val two: Real = 2.0
    val N: Real = x.size

    /*
    {
      val out = new File("/tmp/action.csv")
      val writer = new PrintWriter(out)
      writer.println("lambda,s_x,L")
      for {
        lp <- (0.02 until(1.0, step = 0.01))
        sx <- (0.01 until(0.5, step = 0.005))
      } {
        val g = (1 - lp) / (1 + lp)
        val L = -(diag - g * distWeights.map {
          case (dist, weight) =>
            pow(lp, dist.v) * weight
        }.reduce {
          _ + _
        }) / (2 * sx * sx) + N.v * log(lp) / 2 - N.v * log(sx)
        writer.println(s"$lp,$sx,$L")
      }
      writer.close()
    }
    */

   val optimizer = new Adam(0.1)
  //  val optimizer = new SGDMomentum(mass = 10, lr = 0.1)
    // val optimizer = new BlockLBFGS(histSize = 10, learningRate = 0.1)
    val lambda_odds = Param(0.0)
    val lambda_e = logistic(lambda_odds)
    val s_x_log = Param(-1.0)
    val s_x_e = exp(s_x_log)
    val interpreter = new OptimizingInterpreter(optimizer)
    assert((0 until 10).exists { _ =>
      for {_ <- 0 until 1000} {
        val lambda = interpreter.eval(lambda_e)
        val s_x = interpreter.eval(s_x_e)
        val g = (one - lambda) / (one + lambda)
        val L = -(Value(diag) - g * distWeights.map {
          case (dist, weight) =>
            pow(lambda, dist) * weight
        }.reduce {
          _ + _
        }) / (two * s_x * s_x) + N * log(lambda) / two - N * log(s_x)
        L.dv(1.0)
        interpreter.reset()
      }
      val lambda = interpreter.eval(lambda_e)
      val s_x = interpreter.eval(s_x_e)
      val s_z_v = s_x.v * math.sqrt(lambda.v + 1.0 / lambda.v - 2.0)
      println(s"Lambda: ${lambda.v}, S_x: ${s_x.v}, S_z: ${s_z_v}")
      math.abs(s_x.v - 0.2) < 0.02 && math.abs(s_z_v - 0.1) < 0.02
    })
  }

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
      (x, AutoRegressive(guide, alphaParam), (z, mu, y))
    }

    val volParam = exp(Param(-1.0, "s_z"))
    val errParam = exp(Param(-1.0, "s_x"))
    val model = infer {
      val last_hidden = dataWithGuides.foldLeft(Value(0.0)) {
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