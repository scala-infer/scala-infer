package scappla

import org.scalatest.FlatSpec

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

  /*
  it should "allow a model to be specified" in {

    val sprinkle = infer[Boolean, Boolean] {
      (rain: Boolean) =>
        if (rain) {
          sample(Bernoulli(0.01))
        } else {
          sample(Bernoulli(0.4))
        }
    }

    val model = infer {

      val rain = sample(Bernoulli(0.2))
      val sprinkled = sample(sprinkle(rain))

      val p_wet = (rain, sprinkled) match {
        case (true, true) => 0.99
        case (false, true) => 0.9
        case (true,  false) => 0.8
        case (false, false) => 0.0
      }

      // bind model to data / add observation
      observe(Bernoulli(p_wet), true)

      // return quantity we're interested in
      rain
    }

    val n_rain = Range(0, 10000).map { _ =>
      sample(model)
    }.count(identity)

    println(s"Expected number of rainy days: ${n_rain / 10000.0}")
  }
  */

  it should "recover prior" in {
    val inferred = new Distribution[Boolean] {

      val post = new BBVI("post")

      override def sample(): Variable[Boolean] = {
        sampleVariable(Bernoulli(0.2), post)
      }

      override def score(a: Boolean): Score = ???
    }

    val N = 100
    val n_hits = Range(0, N).map { _ =>
      sample(inferred)
    }.count(identity)

    val p_expected = 0.2
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_hits) < 3 * math.sqrt(n_expected))
  }

  /*
  it should "allow a model to be executed" in {
    val inferred = new Distribution[Boolean] {

      val rainGuide = new BBVI("rain")
      val sprinkleInRainGuide = new BBVI("sir")
      val sprinkleNoRainGuide = new BBVI("snr")

      override def sample(): Variable[Boolean] = {
        val rainVar = sampleVariable(Bernoulli(0.2), rainGuide)
        val rain = rainVar.get

        val sprinkledVar = if (rain)
          sampleVariable(Bernoulli(0.01), sprinkleInRainGuide)
        else
          sampleVariable(Bernoulli(0.4), sprinkleNoRainGuide)
        val sprinkled = sprinkledVar.get
        rainVar.addRef(sprinkledVar.score)

        val p_wet = (rain, sprinkled) match {
          case (true, true) => 0.99
          case (false, true) => 0.9
          case (true,  false) => 0.8
          case (false, false) => 0.001
        }

        val obScore = observeImpl(Bernoulli(p_wet), true).buffer
        sprinkledVar.addRef(obScore)
        rainVar.addRef(obScore)
        new Variable[Boolean] {

          import DValue._

          override def get: Boolean =
            rain

          override def score: Score = {
            rainVar.score + sprinkledVar.score + obScore
          }

          override def complete(): Unit = {
            obScore.complete()
            sprinkledVar.complete()
            rainVar.complete()
          }
        }
      }

      override def score(a: Boolean): Score = ???
    }

    val N = 1000000
    val startTime = System.currentTimeMillis()
    val n_rain = Range(0, N).map { i =>
//      println("")
      if (i % 10000 == 0)
        println(s"${inferred.rainGuide.p.v}")
      sample(inferred)
    }.count(identity)
    val endTime = System.currentTimeMillis()
    println(s"time: ${endTime - startTime} millis => ${(endTime - startTime) * 1000.0 / N} mus / iter")
    println(s"  p(rain): ${inferred.rainGuide.p.v}")

    // See Wikipedia
    // P(rain = true | grass is wet) = 35.77 %

    val p_expected = 0.3577
    val n_expected = p_expected * N
    assert(math.abs(N * p_expected - n_rain) < 3 * math.sqrt(n_expected))
  }
  */

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
