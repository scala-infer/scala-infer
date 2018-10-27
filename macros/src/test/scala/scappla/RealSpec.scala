package scappla

import org.scalatest.FlatSpec
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, ReparamGuide}
import scappla.optimization.{SGD, SGDMomentum}

import scala.util.Random

class RealSpec extends FlatSpec {

  import Functions._

/*
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

    val N = 10000
    // burn in
    for {_ <- 0 to N} {
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
*/

  it should "recover prior" in {
    val inferred = new Model[Boolean] {

      val optimizer = new SGD()
      // p = 1 / (1 + exp(-x)) => x = -log(1 / p - 1)
      val p_guide = sigmoid(optimizer.param(0.0, 10.0))
      //      val p_guide = optimizer.param(0.4)
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

    val sprinkle: Variable[Boolean] => Variable[Boolean] = rainVar => {
      val Variable(rain, node) = rainVar

      val sprinkledVar = if (rain)
        sprinkleInRainGuide.sample(Bernoulli(0.01))
      else
        sprinkleNoRainGuide.sample(Bernoulli(0.4))
      node.addVariable(sprinkledVar.node.modelScore, sprinkledVar.node.guideScore)

      sprinkledVar
    }

    val inferred = new Model[Boolean] {

      override def sample(): Variable[Boolean] = {
        val rainVar = rainGuide.sample(Bernoulli(0.2))
        val Variable(rain, rainNode) = rainVar

        val Variable(sprinkled, sprinkledNode) = sprinkle(rainVar)

        val p_wet = (rain, sprinkled) match {
          case (true, true) => 0.99
          case (false, true) => 0.9
          case (true, false) => 0.8
          case (false, false) => 0.001
        }

        val observation = observeImpl(Bernoulli(p_wet), true)
        sprinkledNode.addObservation(observation.score)
        rainNode.addObservation(observation.score)
        Variable[Boolean](rain, new BayesNode {

          import Real._

          override val modelScore: Score = {
            rainNode.modelScore + sprinkledNode.modelScore + observation.score
          }

          override val guideScore: Score = {
            rainNode.guideScore + sprinkledNode.guideScore
          }

          override def addObservation(score: Score): Unit = {
            rainNode.addObservation(score)
          }

          override def addVariable(modelScore: Score, guideScore: Score): Unit = {
            rainNode.addVariable(modelScore, guideScore)
          }

          override def complete(): Unit = {
            observation.complete()
            sprinkledNode.complete()
            rainNode.complete()
          }
        })
      }
    }

    for {_ <- 0 to 10000} {
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

/*
  it should "reparametrize doubles" in {
    val sgd = new SGD()
    val muGuide = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

    val model = infer {
      val mu = sample(Normal(0.0, 1.0), muGuide)

      observe(Normal(mu, DValue.toConstant(1.0)), DValue.toConstant(2.0))

      mu
    }

    // warm up
    Range(0, 10000).foreach { i =>
      sample(model)
    }

    val N = 10000
    val (total_x, total_xx) = Range(0, N).map { i =>
      sample(model).v
    }.foldLeft((0.0, 0.0)) { case ((sum_x, sum_xx), x) =>
      (sum_x + x, sum_xx + x * x)
    }
    val avg_mu = total_x / N
    val var_mu = total_xx / N - avg_mu * avg_mu
    println(s"Avg mu: ${avg_mu} (${math.sqrt(var_mu)})")
  }
*/

  it should "use the reparametrization gradient" in {

    val data = (0 until 100).map { i =>
      (i.toDouble, 3.0 * i.toDouble + 1.0 + Random.nextGaussian() * 0.2)
    }

    val inferred = new Model[Real] {

      val sgd = new SGD()

      val muGuide = ReparamGuide(Normal(
        sgd.param(0.0, 1.0),
        exp(sgd.param(0.0, 1.0))
      ))

      override def sample(): Variable[Real] = {

        import Real._

        val Variable(mu, muNode) = muGuide.sample(Normal(0.0, 1.0))

        val sigma: Real = Real(1.0)
        val observation: Observation = observeImpl(Normal(mu, sigma), Real(2.0))

        Variable(mu, new BayesNode {

          override def modelScore: Score =
            muNode.modelScore + observation.score

          override def guideScore: Score =
            muNode.guideScore

          override def addObservation(score: Score): Unit =
            muNode.addObservation(score)

          override def addVariable(modelScore: Score, guideScore: Score): Unit =
            muNode.addVariable(modelScore, guideScore)

          override def complete(): Unit = {
            observation.complete()
            muNode.complete()
          }
        })
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

    println(s"Avg mu: ${avg_mu} (${math.sqrt(var_mu)})")
  }

/*
  it should "allow linear regression to be specified" in {
    import DValue._

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

    val sgd = new SGDMomentum(mass = 100)
    val aPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val b1Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val b2Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
    val sPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

    val model = infer {
      val a = sample(Normal(0.0, 1.0), aPost)
      val b1 = sample(Normal(0.0, 1.0), b1Post)
      val b2 = sample(Normal(0.0, 1.0), b2Post)
      val err = exp(sample(Normal(0.0, 1.0), sPost))

      val cb: (((Double, Double), Double)) => Unit = {
        entry: ((Double, Double), Double) =>
          val ((x1, x2), y) = entry
          observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
      }
      data.foreach[Unit](cb)

      (a, b1, b2, err)
    }

    // warm up
    Range(0, 10000).foreach { i =>
      sample(model)
    }

    // print some samples
    Range(0, 10).foreach { i =>
      val l = sample(model)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  ${values}")
    }
  }
*/

  it should "find max likelihood for linear regression without macros" in {

    val rng = new Random(123456789L)
    val data = {
      val alpha = 1.0
      val beta = (1.0, 2.5)
      val sigma = 1.0

      for {_ <- 0 until 1000} yield {
        val X = (rng.nextGaussian(), 0.2 * rng.nextGaussian())
        val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + rng.nextGaussian() * sigma
        (X, Y)
      }
    }

    val lr = 1000.0 / (data.size + 1)

    // find MAP
    val sgd = new SGDMomentum(mass = 100)
    val aPost = Normal(sgd.param(0.0, lr, Some("a-m")), exp(sgd.param(0.0, lr, Some("a-s"))))
    val b1Post = Normal(sgd.param(0.0, lr, Some("b1-m")), exp(sgd.param(0.0, lr, Some("b1-s"))))
    val b2Post = Normal(sgd.param(0.0, lr, Some("b2-m")), exp(sgd.param(0.0, lr, Some("b2-s"))))
    val sPost = Normal(sgd.param(0.0, lr, Some("e-m")), exp(sgd.param(0.0, lr, Some("e-s"))))

    val model = new Model[(Real, Real, Real, Real)] {

      val aGuide = ReparamGuide(aPost)
      val b1Guide = ReparamGuide(b1Post)
      val b2Guide = ReparamGuide(b2Post)
      val sGuide = ReparamGuide(sPost)

      override def sample(): Variable[(Real, Real, Real, Real)] = {
        val aVar = aGuide.sample(Normal(0.0, 1.0))
        val a = aVar.get
        val b1Var = b1Guide.sample(Normal(0.0, 1.0))
        val b1 = b1Var.get
        val b2Var = b2Guide.sample(Normal(0.0, 1.0))
        val b2 = b2Var.get
        val sDraw = sGuide.sample(Normal(0.0, 1.0))
        val err = exp(sDraw.get)

        val cb : Variable[((Double, Double), Double)] => Variable[Unit] = {
          entry =>
            import Real._
            val Variable(((x1, x2), y), node) = entry
            val observation = observeImpl(Normal(a + x1.const * b1 + x2.const * b2, err), y.const)
            node.addObservation(observation.score)
            Variable(Unit, new BayesNode {

              override def modelScore: Score = 0.0

              override def guideScore: Score = 0.0

              override def addObservation(score: Score): Unit = {}

              override def addVariable(modelScore: Score, guideScore: Score): Unit = {}

              override def complete(): Unit = {}
            })
        }
        val wrappedCb =
          new Function[((Double, Double), Double), Unit] with Completeable {

            var modelScore: Real = 0.0
            var nodes: List[BayesNode] = Nil

            override def apply(entry: ((Double, Double), Double)): Unit = {
              val result = cb.apply(Variable(entry, ConstantNode))
              nodes = result.node :: nodes
            }

            override def complete(): Unit = {
              nodes.reverse.foreach(_.complete())
            }
          }
        data.foreach(wrappedCb)

        Variable((a, b1, b2, err), new BayesNode {

          import scappla.Real._

          override val modelScore: Score = wrappedCb.modelScore

          override def guideScore: Score = Real(0.0)

          override def addObservation(score: Score): Unit = {}

          override def addVariable(modelScore: Score, guideScore: Score): Unit = {}

          override def complete(): Unit = {
            wrappedCb.complete()
            sDraw.node.complete()
            b2Var.node.complete()
            b1Var.node.complete()
            aVar.node.complete()
          }
        })
      }
    }

    for {_ <- 0 until 10} {
      // warm up
      Range(0, 100).foreach { i =>
        sample(model)
      }

      println(s"  A post: ${aPost.mu.v} (${aPost.sigma.v})")
      println(s" B1 post: ${b1Post.mu.v} (${b1Post.sigma.v})")
      println(s" B2 post: ${b2Post.mu.v} (${b2Post.sigma.v})")
      println(s"  E post: ${sPost.mu.v} (${sPost.sigma.v})")
    }

    // print some samples
    Range(0, 10).foreach { i =>
      val l = sample(model)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  ${values}")
    }

  }


}
