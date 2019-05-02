package scappla.app

import scappla.Functions.exp
import scappla.distributions.Normal
import scappla.guides.ReparamGuide
import scappla.optimization.SGDMomentum
import scappla._
import scappla.OptimizingInterpreter

import scala.util.Random

object TestLinearRegression extends App {

  val data = {
    val alpha = 1.0
    val sigma = 1.0
    val beta = (1.0, 2.5)

    for {_ <- 0 until 100} yield {
      val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
      val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
      (X, Y)
    }
  }

  val aPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
  val b1Post = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
  val b2Post = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
  val sPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

  val model = infer {
    val a = sample(Normal(0.0, 1.0), aPost)
    val b1 = sample(Normal(0.0, 1.0), b1Post)
    val b2 = sample(Normal(0.0, 1.0), b2Post)
    val err = exp(sample(Normal(0.0, 1.0), sPost))

    val cb = {
      entry: ((Double, Double), Double) =>
        val ((x1, x2), y) = entry
        observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
    }
    data.foreach[Unit](cb)

    (a, b1, b2, err)
  }

  val sgd = new SGDMomentum(mass = 100, lr = 1.0)
  val interp = new OptimizingInterpreter(sgd)

  // warm up
  Range(0, 10000).foreach { i =>
    interp.reset()
    model.sample(interp)
  }

  // print some samples
  Range(0, 10).foreach { i =>
    interp.reset()
    val l = model.sample(interp)
    val values = (l._1.v, l._2.v, l._3.v, l._4.v)
    println(s"  ${values}")
  }

}
