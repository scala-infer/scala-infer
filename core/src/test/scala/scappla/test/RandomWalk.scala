package scappla.test

import org.scalatest.FlatSpec
import scala.util.Random

import scappla._
import scappla.Functions._
import scappla.distributions.Normal
import scappla.guides.ReparamGuide
import scappla.optimization.Adam
import scappla.guides.Guide
import scappla.guides.AutoRegressive

class RandomWalkSpec extends FlatSpec {

  it should "fit volatility" in {

    val hidden = (0 until 200).scanLeft(0.0) { case (v, _) =>
      v + Random.nextGaussian() * 0.1
    }
    val data = hidden.map { v =>
      (v, v + 0.2 * Random.nextGaussian())
    }

    val volGuide = ReparamGuide(Normal(Param(-2.0), exp(Param(0.0))))
    val errGuide = ReparamGuide(Normal(Param(-2.0), exp(Param(0.0))))
    val lambdaGuide = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

    val dataGuideErr = exp(Param(-2.0))
    val dataWithGuides = data.map { v =>
      val mu = Param(0.0)
      val guide = ReparamGuide(Normal(mu, dataGuideErr))
      (v, AutoRegressive(guide, 1.0), mu)
    }

    val volParam = exp(Param(0.0))
    val errParam = exp(Param(0.0))
    // val lambdaParam = logistic(Param(0.0))
    val model = infer {
      // val vol = exp(sample(Normal(-2.0, 1.0), volGuide))
      // val err = exp(sample(Normal(-2.0, 1.0), errGuide))
      val lambda = logistic(sample(Normal(0.0, 1.0), lambdaGuide))

      val ultimate = dataWithGuides.foldLeft((Value(0.0), Value(0.0))) {
        case ((prev, acc), ((_, value), ar, _)) =>
          // prior: (z_i - z_{i-1})^2 / 2 vol^2
          // posterior: (z_i - (1 - \lambda) * \sum_{j < i}\lambda^{j - i - 1} z_j)^2 / 2 \hat vol^2
          val hidden: Real = sample(Normal(prev, volParam), ar.guide(acc))
          // observation: (z_i - x_i)^2 / 2 err^2
          observe(Normal(hidden, errParam), value: Real)
          (hidden, (1.0 - lambda) * hidden + lambda * acc)
      }
      lambda
      // (vol, err, lambda)
      // (vol, err)
    }

    val interpreter = new OptimizingInterpreter(new Adam(0.001))
    for { iter <- 0 until 10000 } {
      // val (vol, err, lambda) = model.sample(interpreter)
      val lambda = model.sample(interpreter)
      // val (vol, err) = model.sample(interpreter)
      if (iter % 100 == 0) {
        val vol = interpreter.eval(volParam)
        val err = interpreter.eval(errParam)
        val dErr = interpreter.eval(dataGuideErr)
        // val lambda = interpreter.eval(lambdaParam)
        // println(s"${iter}: Vol ${vol}, Err: ${err}, Lambda: ${lambda}")
        // println(s"RANDOMWALK: ${iter}, ${vol.v}, ${err.v}, ${lambda.v}")
        println(s"RANDOMWALK: ${iter}, ${vol.v}, ${err.v}, ${lambda.v}, ${dErr.v}")
      }
      interpreter.reset()
    }
    val lambda = model.sample(interpreter)
    dataWithGuides.foldLeft(0.0){ case (sum, ((v, o), _, mu)) =>
      val muv = interpreter.eval(mu).v
      // val result = muv
      println(s"HIDDEN: ${v}, ${muv + (1.0 - lambda.v) * sum}, ${o}")
      sum + muv
    }

  }
}