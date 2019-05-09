package scappla.test

import org.scalatest.FlatSpec
import scappla._
import scappla.Functions._
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, ReparamGuide}
import scappla.optimization.Adam
import scappla.tensor._

import scala.util.Random

class MacrosSpec extends FlatSpec {

  it should "allow a model to be specified" in {

//    val sgd = new SGD()
    val inRain = BBVIGuide(Bernoulli(logistic(Param(0.0))))
    val noRain = BBVIGuide(Bernoulli(logistic(Param(0.0))))

    val rainPost = BBVIGuide(Bernoulli(logistic(Param(0.0))))

    val model = infer {

      val sprinkle = {
        rain: Boolean =>
          if (rain) {
            sample(Bernoulli(0.01), inRain)
          } else {
            sample(Bernoulli(0.4), noRain)
          }
      }

      val rain = sample(Bernoulli(0.2), rainPost)
      val sprinkled = sprinkle(rain)

      val p_wet = (rain, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true, false) => 0.8
        case (false, false) => 0.001
      }

      // bind model to data / add observation
      observe(Bernoulli(p_wet), true)

      // return quantity we're interested in
      rain
    }

    val sgd = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(sgd)

    val N = 10000
    // burn in
    for {_ <- 0 to N} {
      interpreter.reset()
      model.sample(interpreter)
    }

    // measure
    val n_rain = Range(0, N).map { _ =>
      interpreter.reset()
      model.sample(interpreter)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %
    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
  }

  it should "reparametrize doubles" in {
    val muGuide = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

    val model: Model[Real] = infer {
      val mu = sample(Normal(0.0, 1.0), muGuide)

      observe(Normal(mu, 1.0), 2.0: Real)

      mu
    }

    val sgd = new Adam()
    val interpreter = new OptimizingInterpreter(sgd)

    // warm up
    Range(0, 10000).foreach { i =>
      interpreter.reset()
      model.sample(interpreter)
    }

    val N = 10000
    val (total_x, total_xx) = Range(0, N).map { i =>
      interpreter.reset()
      model.sample(interpreter).v
    }.foldLeft((0.0, 0.0)) { case ((sum_x, sum_xx), x) =>
      (sum_x + x, sum_xx + x * x)
    }
    val avg_mu = total_x / N
    val var_mu = total_xx / N - avg_mu * avg_mu
    println(s"Avg mu: $avg_mu (${math.sqrt(var_mu)})")
  }

  it should "allow linear regression to be specified" in {

    val data = {
      val alpha = 1.0
      val sigma = 1.0
      val beta = (1.0, 2.5)

      for {_ <- 0 until 1000} yield {
        val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
        val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
        (X, Y)
      }
    }

    case class NormalParams(mu: Expr[Double, Unit], sigma: Expr[Double, Unit])

    def normalParams() = {
      NormalParams(Param(0.0), exp(Param(0.0)))
    }

    val aParam = normalParams()
    val aPost = ReparamGuide(Normal(aParam.mu, aParam.sigma))

    val b1Param = normalParams()
    val b1Post = ReparamGuide(Normal(b1Param.mu, b1Param.sigma))

    val b2Param = normalParams()
    val b2Post = ReparamGuide(Normal(b2Param.mu, b2Param.sigma))

    val sParam = normalParams()
    val sPost = ReparamGuide(Normal(sParam.mu, sParam.sigma))

    val model = infer {
      val a = sample(Normal(0.0, 1.0), aPost)
      val b1 = sample(Normal(0.0, 1.0), b1Post)
      val b2 = sample(Normal(0.0, 1.0), b2Post)
      val err = exp(sample(Normal(0.0, 1.0), sPost))

      data.foreach[Unit] {
        case ((x1, x2), y) =>
          observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
      }

      (a, b1, b2, err)
    }

//    val sgd = new SGDMomentum(mass = 100)
    val sgd = new Adam(0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(sgd)

    // warm up
    Range(0, 1000).foreach { i =>
      interpreter.reset()
      model.sample(interpreter)
    }

    def asString(params: NormalParams, name: String): String = {
      s"$name: ${interpreter.eval(params.mu).v} (${interpreter.eval(params.sigma).v})"
    }

    println(asString(aParam, "  A post: "))
    println(asString(b1Param, " B1 post: "))
    println(asString(b2Param, " B2 post: "))
    println(asString(sParam, "  E post: "))

    // print some samples
    Range(0, 10).foreach { i =>
      interpreter.reset()
      val l = model.sample(interpreter)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  $values")
    }
  }

  it should "allow linear regression to be specified using tensors" in {
    import Tensor._

    val N = 1000

    case class Batch(size: Int) extends Dim[Batch]
    val batch = Batch(N)

    val (x1, x2, y) = {
      val alpha = 1.0
      val sigma = 1.0
      val beta = (1.0, 2.5)

      val x1 = new Array[Float](N)
      val x2 = new Array[Float](N)
      val y = new Array[Float](N)
      for {i <- 0 until 1000} {
        val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
        val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
        x1(i) = X._1.toFloat
        x2(i) = X._2.toFloat
        y(i) = Y.toFloat
      }
      (
          Value(ArrayTensor(Seq(N), x1), batch),
          Value(ArrayTensor(Seq(N), x2), batch),
          Value(ArrayTensor(Seq(N), y), batch)
      )
    }

    case class NormalParams(mu: Expr[Double, Unit], sigma: Expr[Double, Unit])

    def normalParams() = {
      NormalParams(Param(0.0), exp(Param(0.0)))
    }

    val aParam = normalParams()
    val aPost = ReparamGuide(Normal(aParam.mu, aParam.sigma))

    val b1Param = normalParams()
    val b1Post = ReparamGuide(Normal(b1Param.mu, b1Param.sigma))

    val b2Param = normalParams()
    val b2Post = ReparamGuide(Normal(b2Param.mu, b2Param.sigma))

    val sParam = normalParams()
    val sPost = ReparamGuide(Normal(sParam.mu, sParam.sigma))

    import Value._

    val model = infer {
      val a = sample(Normal(0.0, 1.0), aPost)
      val b1 = sample(Normal(0.0, 1.0), b1Post)
      val b2 = sample(Normal(0.0, 1.0), b2Post)
      val err = exp(sample(Normal(0.0, 1.0), sPost))

      val mu = broadcast(a, batch) +
            broadcast(b1, batch) * x1 +
            broadcast(b2, batch) * x2
      val sigma = broadcast(err, batch)
      observe(Normal[ArrayTensor, Batch](mu, sigma), y.const)

      (a, b1, b2, err)
    }

    val sgd = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(sgd)

    // warm up
    Range(0, 100).foreach { i =>
      interpreter.reset()
      model.sample(interpreter)
    }

    def asString(params: NormalParams, name: String): String = {
      s"$name: ${interpreter.eval(params.mu).v} (${interpreter.eval(params.sigma).v})"
    }

    println(asString(aParam, "  A post: "))
    println(asString(b1Param, " B1 post: "))
    println(asString(b2Param, " B2 post: "))
    println(asString(sParam, "  E post: "))

    // print some samples
    Range(0, 10).foreach { i =>
      interpreter.reset()
      val l = model.sample(interpreter)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s" (Tensor LR) $values")
    }
  }

  it should "allow mixing discrete/continuous variables" in {
    val data = {
      val p = 0.75
      val mus = Seq(-0.5, 1.2)
      val sigma = 0.2

      for {_ <- 0 until 500} yield {
        if (Random.nextDouble() < p) {
          Random.nextGaussian() * sigma + mus(0)
        } else {
          Random.nextGaussian() * sigma + mus(1)
        }
      }
    }

    val pPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
    val mu1Post = ReparamGuide(Normal(Param(-1.0), exp(Param(-1.0))))
    val mu2Post = ReparamGuide(Normal(Param(1.0), exp(Param(-1.0))))
    val sigmaPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

    val dataWithDist = data.map { datum =>
      (datum, BBVIGuide(Bernoulli(logistic(Param(0.0)))))
    }

    import Value._

    val model = infer {
      val p = logistic(sample(Normal(0.0, 1.0), pPost))
      val mu1 = sample(Normal(0.0, 1.0), mu1Post)
      val mu2 = sample(Normal(0.0, 1.0), mu2Post)
      val sigma = exp(sample(Normal(0.0, 1.0), sigmaPost))

      dataWithDist.foreach[Unit] {
        case (x, guide) =>
          val i = sample(Bernoulli(p), guide)
          if (i) {
            observe(Normal(mu1, sigma), x.const)
          } else {
            observe(Normal(mu2, sigma), x.const)
          }
      }

      (p, mu1, mu2, sigma)
    }

    val sgd = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(sgd)

    // warm up
    Range(0, 1000).foreach { i =>
      interpreter.reset()
      model.sample(interpreter)
    }
    val N = 5000
    val startTime = System.currentTimeMillis()
    Range(0, N).foreach { i =>
      interpreter.reset()
      model.sample(interpreter)
    }
    val endTime = System.currentTimeMillis()
    println(s"time: ${endTime - startTime} millis => ${(endTime - startTime) * 1000.0 / N} mus / iter")

    // print some samples
    Range(0, 10).foreach { i =>
      interpreter.reset()
      val l = model.sample(interpreter)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  $values")
    }
  }

}
