package scappla

import org.scalatest.FlatSpec

import scala.util.Random

class DValueSpec extends FlatSpec {

  import Functions._

  "The ad macro" should "compute the backward gradient of a polynomial" in {
    val fn = autodiff { (z: Double) => z + z * z * z }

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 10.0)

    value.dv(1.0)

    assert(variable.grad == 13.0)
  }

  it should "compute the backward gradient of the log" in {
    val fn = autodiff { (z: Double) => z * log(z) }

    val variable = new DVariable(2.0)

    val value: DValue[Double] = fn(variable)
    assert(value.v == 2.0 * scala.math.log(2.0))

    value.dv(1.0)

    assert(variable.grad == scala.math.log(2.0) + 1.0)

  }

  it should "compute gradient of pow for base" in {
    val fn = autodiff { (z: Double) => z + pow(z, 3.0) }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)

    val exact: Double => Double = z => 1.0 + 3 * pow(z, 2.0)
    assert(variable.grad == exact(0.5))
  }

  it should "compute gradient of Math.pow for exponent" in {
    val fn = autodiff { (z: Double) =>
      z + pow(2.0, z)
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)

    val exact: Double => Double =
      z => 1.0 + log(2.0) * pow(2.0, z)
    assert(variable.grad == exact(0.5))
  }

  it should "compute gradient with a block in body" in {
    val fn = autodiff {
      (z: Double) => {
        val x = z * z * z
        x
      }
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = fn(variable)
    value.dv(1.0)

    assert(variable.grad == 0.75)
  }

  it should "compose gradients" in {
    val square = autodiff {
      (x: Double) => x * x
    }
    val plus_x = autodiff {
      (x: Double) => x + square(x)
    }

    val variable = new DVariable(0.5)
    val value: DValue[Double] = plus_x(variable)
    value.dv(1.0)

    assert(variable.grad == 2.0)
  }

  it should "allow a model to be specified" in {

    val sgd = new SGD()
    val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
    val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val sprinkle = infer {
      rain: Boolean =>
        if (rain) {
          sample(Bernoulli(0.01), inRain)
        } else {
          sample(Bernoulli(0.4), noRain)
        }
    }

    val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

    val model = infer {

      val rain = sample(Bernoulli(0.2), rainPost)
      val sprinkled = sample(sprinkle(rain))

      val p_wet = (rain, sprinkled) match {
        case (true,  true)  => 0.99
        case (false, true)  => 0.9
        case (true,  false) => 0.8
        case (false, false) => 0.001
      }

      // bind model to data / add observation
      observe(Bernoulli(p_wet), true)

      // return quantity we're interested in
      rain
    }

    //val N = 10000
    val N = 5
    // burn in
    for { _ <- 0 to N } {
      sample(model)
    }

    // measure
    val n_rain = Range(0, N).map { _ =>
      sample(model)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %
    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
  }

  it should "recover prior" in {
    val inferred = new Model[Boolean] {

      val optimizer = new SGD()
      // p = 1 / (1 + exp(-x)) => x = -log(1 / p - 1)
      val p_guide = sigmoid(optimizer.param[Double](0.0, 10.0))
//      val p_guide = optimizer.param[Double](0.4)
      val guide = BBVIGuide(Bernoulli(p_guide))

      override def sample(): Variable[Boolean] = {
        guide.sample(Bernoulli(0.2))
      }
    }

    val N = 10000
    Range(0, N).foreach { _ =>
      sample(inferred)
    }
    val n_hits = Range(0, N).map { _ =>
      sample(inferred)
    }.count(identity)

    val p_expected = 0.2
    val n_expected = p_expected * N
    println(s"N hits: ${n_hits} (expected: ${n_expected}); p_guide: ${inferred.p_guide.v}")
    assert(math.abs(N * p_expected - n_hits) < 3 * math.sqrt(n_expected))
  }

  it should "allow a discrete model to be executed" in {

    val sgd = new SGD()
    val sprinkleInRainGuide = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0, 10.0))))
    val sprinkleNoRainGuide = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0, 10.0))))

    val rainGuide = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0, 10.0))))

    val sprinkle = new Model1[Boolean, Boolean] {

      override def apply(rain: Boolean): Model[Boolean] = new Model[Boolean] {

        private var rainVar: Variable[_] = null

        override def withDeps(rainVar: Variable[_]) = {
          this.rainVar = rainVar
          this
        }

        override def sample() = {
          val sprinkledVar = if (rain)
            sprinkleInRainGuide.sample(Bernoulli(0.01))
          else
            sprinkleNoRainGuide.sample(Bernoulli(0.4))
          rainVar.addVariable(sprinkledVar.modelScore, sprinkledVar.guideScore)

          sprinkledVar
        }
      }
    }

    val inferred = new Model[Boolean] {

      override def sample(): Variable[Boolean] = {
        val rainVar = rainGuide.sample(Bernoulli(0.2))
        val rain = rainVar.get

        val sprinkledVar = sprinkle(rain).withDeps(rainVar).sample()
        val sprinkled = sprinkledVar.get

        val p_wet = (rain, sprinkled) match {
          case (true, true) => 0.99
          case (false, true) => 0.9
          case (true,  false) => 0.8
          case (false, false) => 0.001
        }

        val observation = observeImpl(Bernoulli(p_wet), true)
        sprinkledVar.addObservation(observation.score)
        rainVar.addObservation(observation.score)
        new Variable[Boolean] {

          import DValue._

          override val get: Boolean =
            rain

          override val modelScore: Score = {
            rainVar.modelScore + sprinkledVar.modelScore + observation.score
          }

          override val guideScore: Score = {
            rainVar.guideScore + sprinkledVar.guideScore
          }

          override def addObservation(score: Score): Unit = {
            rainVar.addObservation(score)
          }

          override def addVariable(modelScore: Score, guideScore: Score): Unit = {
            rainVar.addVariable(modelScore, guideScore)
          }

          override def complete(): Unit = {
            observation.complete()
            sprinkledVar.complete()
            rainVar.complete()
          }
        }
      }
    }

    for { _ <- 0 to 10000 } {
      sample(inferred)
    }

    val N = 10000
    val startTime = System.currentTimeMillis()
    val n_rain = Range(0, N).map { i =>
      sample(inferred)
    }.count(identity)
    val endTime = System.currentTimeMillis()
    println(s"time: ${endTime - startTime} millis => ${(endTime - startTime) * 1000.0 / N} mus / iter")

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %
    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
  }

  it should "use the reparametrization gradient" in {

    val data = (0 until 100).map { i =>
      (i.toDouble, 3.0 * i.toDouble + 1.0 + Random.nextGaussian() * 0.2)
    }

    val inferred = new Model[DValue[Double]] {

      val sgd = new SGD()

      val muGuide = ReparamGuide(Normal(
        sgd.param(0.0, 1.0),
        exp(sgd.param(0.0, 1.0))
      ))

      override def sample(): Variable[DValue[Double]] = new Variable[DValue[Double]] {

        import DValue._

        val muVar = muGuide.sample(Normal(0.0, 1.0))

        private val sigma: Variable[DValue[Double]] = Variable.toConstant(DValue.toConstant(1.0))
        private val observation : Observation = observeImpl(Normal(muVar, sigma), DValue.toConstant(2.0))

        override def get: DValue[Double] =
          muVar.get

        override def modelScore: Score =
          muVar.modelScore + observation.score

        override def guideScore: Score =
          muVar.guideScore

        override def addObservation(score: Score): Unit =
          muVar.addObservation(score)

        override def addVariable(modelScore: Score, guideScore: Score): Unit =
          muVar.addVariable(modelScore, guideScore)

        override def complete(): Unit = {
          observation.complete()
          muVar.complete()
        }
      }
    }

    // warm up
    Range(0, 10000).foreach { i =>
      sample(inferred)
    }

    val N = 10000
    val startTime = System.currentTimeMillis()
    val (total_x, total_xx) = Range(0, N).map { i =>
      sample(inferred).v
    }.foldLeft((0.0, 0.0)) { case ((sum_x, sum_xx), x) =>
      (sum_x + x, sum_xx + x * x)
    }
    val avg_mu = total_x / N
    val var_mu = total_xx / N - avg_mu * avg_mu
    val endTime = System.currentTimeMillis()
    println(s"time: ${endTime - startTime} millis => ${(endTime - startTime) * 1000.0 / N} mus / iter")

    println(s"Avg mu: ${avg_mu} (${math.sqrt(var_mu)}")
  }

  /*
  it should "allow inference by enumeration" in {
    val px = 0.2
    val pty = 0.7
    val pfy = 0.8
    val success = for {
      x <- Bernoulli(px).draw(new Enumeration())
      y <- Bernoulli(if (x) pty else pfy).draw(new Enumeration())
    } yield (x, y)

    val N = 10000
    val n_success = (0 to N)
        .map { _ => sample(success) }
        .groupBy(identity)
        .mapValues(_.size)

    def expected(x: Boolean, y: Boolean) = {
      (if (x) px else 1.0 - px) *
          (if (x) {
            if (y) pty else 1.0 - pty
          } else {
            if (y) pfy else 1.0 - pfy
          })
    }

    for {((x, y), count) <- n_success} {
      val n = expected(x, y) * N
      assert(math.abs(count - n) < 3 * math.sqrt(n))
    }
  }
  */


  /*
  it should "allow a model to be specified" in {

    object guides {
      val sprinkleWhenRain = Bernoulli(0.01)
      val sprinkleWithoutRain = Bernoulli(0.4)
      val rainPosterior = Bernoulli(0.2)
    }

    val sprinkle: Variable[Boolean] => Variable[Boolean] =
      (rain: Variable[Boolean]) => for {
        s <- if (rain.get) {
          sample[Boolean](Bernoulli(0.01), guides.sprinkleWhenRain)
        } else {
          sample[Boolean](Bernoulli(0.4), guides.sprinkleWithoutRain)
        }
      } yield s

    val hasRained = for {
      rain <- sample(Bernoulli(0.2), guides.rainPosterior)
      sprinkledVar = sprinkle(rain)
      sprinkled <- sprinkledVar
      p_wet = (rain.get, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true, false) => 0.8
        case (false, false) => 0.001
      }

      // bind model to data / add observation
      Bernoulli(p_wet).observe(true)
    } yield rain

    val N = 100000
    val n_rain = Range(0, N).map { _ =>
      sample(hasRained)
    }.count(identity)

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %

    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
  }
  */

}
