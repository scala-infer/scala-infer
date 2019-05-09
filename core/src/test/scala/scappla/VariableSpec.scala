package scappla

import org.scalatest.FlatSpec
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.{BBVIGuide, ReparamGuide}
import scappla.optimization.Adam

import scala.util.Random

class VariableSpec extends FlatSpec {

  import Value._
  import Expr._
  import Functions._

  it should "recover prior" in {

    val par = Param(0.0)
    val p_guide = logistic(par)
    val guide = BBVIGuide(Bernoulli(p_guide))

    val inferred = new Model[Boolean] {

      override def sample(interpreter: Interpreter): Boolean = {
        val Variable(value, node) = guide.sample(interpreter, Bernoulli(0.2))
        node.complete()
        value
      }

    }

    val optimizer = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(optimizer)

    val N = 1000
    Range(0, N).foreach { _ =>
      interpreter.reset()
      inferred.sample(interpreter)
    }
    val n_hits = Range(0, N).map { _ =>
      interpreter.reset()
      inferred.sample(interpreter)
    }.count(identity)

    val p_expected = 0.2
    val n_expected = p_expected * N
    println(s"N hits: ${n_hits} (expected: ${n_expected}); p_guide: ${interpreter.eval(p_guide).v}")
    assert(math.abs(N * p_expected - n_hits) < 3 * math.sqrt(n_expected))
  }

  it should "allow a discrete model to be executed" in {

    import Expr._

    val sprinkleInRainGuide = BBVIGuide(Bernoulli(logistic(Param(0.0))))
    val sprinkleNoRainGuide = BBVIGuide(Bernoulli(logistic(Param(0.0))))

    val rainGuide = BBVIGuide(Bernoulli(logistic(Param(0.0))))

    def sprinkle(interpreter: Interpreter, rainVar: Variable[Boolean]) = {
        val Variable(rain, node) = rainVar

        val sprinkledVar = if (rain)
          sprinkleInRainGuide.sample(interpreter, Bernoulli(0.01))
        else
          sprinkleNoRainGuide.sample(interpreter, Bernoulli(0.4))
        node.addVariable(sprinkledVar.node.modelScore, sprinkledVar.node.guideScore)

        sprinkledVar
    }

    val inferred = new Model[Boolean] {

      override def sample(interpreter: Interpreter): Boolean = {
        val rainVar = rainGuide.sample(interpreter, Bernoulli(0.2))
        val Variable(rain, rainNode) = rainVar

        val Variable(sprinkled, sprinkledNode) = sprinkle(interpreter, rainVar)

        val p_wet = (rain, sprinkled) match {
          case (true, true) => 0.99
          case (false, true) => 0.9
          case (true, false) => 0.8
          case (false, false) => 0.001
        }

        val observation = observeImpl(interpreter, Bernoulli(p_wet), true)
        sprinkledNode.addObservation(observation.score)
        rainNode.addObservation(observation.score)

        observation.complete()
        sprinkledNode.complete()
        rainNode.complete()

        rain
      }
    }

    val optimizer = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(optimizer)

    for {_ <- 0 to 10000} {
      interpreter.reset()
      inferred.sample(interpreter)
    }

    val N = 10000
    val startTime = System.currentTimeMillis()
    val n_rain = Range(0, N).map { i =>
      interpreter.reset()
      inferred.sample(interpreter)
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

    val inferred = new Model[Real] {

      val muGuide = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

      override def sample(interpreter: Interpreter): Real = {

        val Variable(mu, muNode) = muGuide.sample(interpreter, Normal(0.0, 1.0))

        val observation: Observation = observeImpl(interpreter, Normal(mu, 1.0), 2.0: Real)

        observation.complete()
        muNode.complete()

        mu
      }
    }

    val sgd = new Adam()
    val interpreter = new OptimizingInterpreter(sgd)

    // warm up
    Range(0, 10000).foreach { i =>
        interpreter.reset()
        inferred.sample(interpreter)
    }

    val N = 10000
    val startTime = System.currentTimeMillis()
    val (total_x, total_xx) = Range(0, N).map { i =>
      interpreter.reset()
      inferred.sample(interpreter).v
    }.foldLeft((0.0, 0.0)) { case ((sum_x, sum_xx), x) =>
      (sum_x + x, sum_xx + x * x)
    }
    val avg_mu = total_x / N
    val var_mu = total_xx / N - avg_mu * avg_mu
    val endTime = System.currentTimeMillis()
    println(s"time: ${endTime - startTime} millis => ${(endTime - startTime) * 1000.0 / N} mus / iter")

    println(s"Avg mu: ${avg_mu} (${math.sqrt(var_mu)})")
  }

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

    val aPost = Normal(Param(0.0, "a-m"), exp(Param(0.0, "a-s")))
    val b1Post = Normal(Param(0.0, "b1-m"), exp(Param(0.0, "b1-s")))
    val b2Post = Normal(Param(0.0, "b2-m"), exp(Param(0.0, "b2-s")))
    val sPost = Normal(Param(0.0, "e-m"), exp(Param(0.0, "e-s")))

    val model = new Model[(Real, Real, Real, Real)] {

      val aGuide = ReparamGuide(aPost)
      val b1Guide = ReparamGuide(b1Post)
      val b2Guide = ReparamGuide(b2Post)
      val sGuide = ReparamGuide(sPost)

      override def sample(interpreter: Interpreter): (Real, Real, Real, Real) = {
        val aVar = aGuide.sample(interpreter, Normal(0.0, 1.0))
        val a = aVar.get.buffer
        val b1Var = b1Guide.sample(interpreter, Normal(0.0, 1.0))
        val b1 = b1Var.get.buffer
        val b2Var = b2Guide.sample(interpreter, Normal(0.0, 1.0))
        val b2 = b2Var.get.buffer
        val sDraw = sGuide.sample(interpreter, Normal(0.0, 1.0))
        val err = exp(sDraw.get).buffer

        val cb: Variable[((Double, Double), Double)] => Variable[Unit] = {
          entry =>
            val Variable(((x1, x2), y), node) = entry
            val observation = observeImpl(interpreter, Normal(a + x1 * b1 + x2 * b2, err), y: Real)
            node.addObservation(observation.score)
            Variable(Unit, new BayesNode {

              override def modelScore: Score = 0.0

              override def guideScore: Score = 0.0

              override def addObservation(score: Score): Unit = {}

              override def addVariable(modelScore: Score, guideScore: Score): Unit = {}

              override def complete(): Unit = {
                observation.complete()
              }
            })
        }
        val wrappedCb =
          new Function[((Double, Double), Double), Unit] with Completeable {

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

        wrappedCb.complete()
        err.complete()
        sDraw.node.complete()
        b2.complete()
        b2Var.node.complete()
        b1.complete()
        b1Var.node.complete()
        a.complete()
        aVar.node.complete()

        (a, b1, b2, err)
      }
    }

    val sgd = new Adam(alpha = 0.1, epsilon = 1e-4)
    val interpreter = new OptimizingInterpreter(sgd)

    for {_ <- 0 until 10} {
      // warm up
      Range(0, 100).foreach { i =>
        interpreter.reset()
        model.sample(interpreter)
      }

      def valueOf(expr: Expr[Double, Unit]): Double = {
        interpreter.eval(expr).v
      }

      println(s"  A post: ${valueOf(aPost.mu)} (${valueOf(aPost.sigma)})")
      println(s" B1 post: ${valueOf(b1Post.mu)} (${valueOf(b1Post.sigma)})")
      println(s" B2 post: ${valueOf(b2Post.mu)} (${valueOf(b2Post.sigma)})")
      println(s"  E post: ${valueOf(sPost.mu)} (${valueOf(sPost.sigma)})")
    }

    // print some samples
    Range(0, 10).foreach { i =>
      interpreter.reset()
      val l = model.sample(interpreter)
      val values = (l._1.v, l._2.v, l._3.v, l._4.v)
      println(s"  ${values}")
    }

  }

}
