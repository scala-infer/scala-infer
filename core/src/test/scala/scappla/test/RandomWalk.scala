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
    val data = hidden.map {
      _ + 0.2 * Random.nextGaussian()
    }

    val volGuide = ReparamGuide(Normal(Param(-2.0), exp(Param(-1.0))))
    val errGuide = ReparamGuide(Normal(Param(-2.0), exp(Param(-1.0))))
    val lambdaGuide = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

    val dataGuideErr = exp(Param(-2.0))
    val dataWithGuides = data.map { v =>
      val guide = ReparamGuide(Normal(Param(0.0), dataGuideErr))
      (v, AutoRegressive(guide, 1.0))
    }

    val model = infer {
      val vol = exp(sample(Normal(-2.0, 0.1), volGuide))
      val err = exp(sample(Normal(-2.0, 0.1), errGuide))
      val lambda = logistic(sample(Normal(0.0, 1.0), lambdaGuide))

      val ultimate = dataWithGuides.foldLeft((Value(0.0))) {
        case (prev, (value, ar)) =>
          val hidden: Real = sample(Normal(prev, vol), ar.guide(prev))
          observe(Normal(hidden, err), value: Real)
          hidden * (1.0 - lambda) + prev * lambda
      }
      (vol, err, lambda)
    }

    val interpreter = new OptimizingInterpreter(new Adam(0.01))
    for { iter <- 0 until 10000 } {
      val (vol, err, lambda) = model.sample(interpreter)
      if (iter % 100 == 0) {
        // val vol = interpreter.eval(volParam).v
        // val err = interpreter.eval(errParam).v
        // val lambda = interpreter.eval(lambdaParam).v
        // println(s"${iter}: Vol ${vol}, Err: ${err}, Lambda: ${lambda}")
        println(s"RANDOMWALK: ${iter}, ${vol.v}, ${err.v}, ${lambda.v}")
      }
      interpreter.reset()
    }

  }
}