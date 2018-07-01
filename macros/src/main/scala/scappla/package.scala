import scala.util.Random
import scala.language.experimental.macros
import scappla.DValue._
import scappla.Functions._
import com.typesafe.scalalogging._

import scala.collection.mutable

package object scappla {

  type Score = DValue[Double]

  def autodiff[A, B](fn: A => B): DFunction1[A, B] =
  macro Macros.autodiff[A, B]

  // API
  /**
   * The outermost "sample" is public - it drives the inference.  Each sample that's obtained
   * is used to further optimize the inferred distribution.  This can be by means of Monte Carlo,
   * or Variational Inference.
   */
  def sample[X](model: Model[X]): X = {
    val variable = model.sample()
    val value = variable.get

    // prepare for next sample - update approximation
    variable.complete()
    value
  }

  /**
    * Sampling in an infer block.
    * This implimentation will be overridden by a macro.
    */
  def sample[X](prior: Distribution[X], posterior: Distribution[X]): X =
    posterior.sample().get

  def infer[X](fn: X): scappla.Model[X] =
    macro Macros.infer[X]

  def infer[Y, X](fn: Y => X): scappla.Model1[Y, X] =
    macro Macros.infer1[Y, X]

  def observe[A](distribution: Distribution[A], value: A): Unit = {}

  // IMPLEMENTATION

  def observeImpl[A](distribution: Distribution[A], value: A): Observation =
    new Observation {

      val score: Buffer[Double] =
        distribution.observe(value).buffer

      override def complete(): Unit = {
        score.dv(1.0)
//        println("completing observation score")
        score.complete()
      }
    }

  def sampleImpl[A](distribution: Model[A]): Variable[A] = {
    distribution.sample()
  }

  trait Sample[A] {

    def get: A

    def score: Score

    def complete(): Unit
  }

  trait Distribution[A] {

    def sample(): Sample[A]

    def observe(a: A): Score
  }

  trait DDistribution[A] extends Distribution[DValue[A]] {

    def reparam_score(a: DValue[A]): Score
  }

  trait Optimizer {

    def param[X: Fractional](initial: X, lr: X): DValue[X]
  }

  class SGD(val debug: Boolean = false) extends Optimizer {

    override def param[X: Fractional](initial: X, lr: X): DValue[X] = {
      val num = implicitly[Fractional[X]]
      import num._
      new DValue[X] {

        private var iter: Int = 0

        private var value: X = initial

        override def v: X = value

        override def dv(dv: X): Unit = {
          iter += 1
          value = value + dv * lr / num.fromInt(iter)
          if (debug) {
            println(s"    SGD $iter: $value ($dv)")
//          new Exception().printStackTrace()
          }
        }
      }
    }
  }

  trait Variable[A] {

    def get: A

    /**
      * Score (log probability) of the variable in the model.  This is equal to the log
      * of the prior probability that the variable takes the value returned by "get".
      */
    def modelScore: Score

    /**
      * Score (log probability) of the variable in the guide.  Equal to the log of the
      * approximate posterior probability that the variable has the value returned by "get".
      */
    def guideScore: Score

    /**
     * Registers a "score" that depends on the value of the variable.  This can be used
     * to compose the Markov Blanket for the variable.  By having the subset of scores in the blanket,
     * Rao-Blackwellization can be carried out to reduce the variance of the gradient estimate.
     * <p>
     * This method just adds a score to the (generative) model, i.e. it corresponds to an observation.
     */
    def addObservation(score: Score): Unit

    /**
      * Registers a "score" that depends on the value of the variable.  This can be used
      * to compose the Markov Blanket for the variable.  By having the subset of scores in the blanket,
      * Rao-Blackwellization can be carried out to reduce the variance of the gradient estimate.
      * <p>
      * This method is intended for adding downstream variables - both model and guide scores
      * are needed.
      */
    def addVariable(modelScore: Score, guideScore: Score): Unit

    /**
      * When the value of the variable is retrieved and it can no longer be
      * exposed to any new dependencies, it should be completed.  In the case of the
      * outer "sample", this means updating the inferred distributions.
      */
    def complete(): Unit
  }

  object Variable extends LazyLogging {

    implicit def toConstant[A](value: A): Variable[A] =
      new Variable[A] {

        override def get: A = value

        override def modelScore: Score = 0.0

        override def guideScore: Score = 0.0

        override def complete(): Unit = {}

        override def addObservation(score: Score): Unit =
          logger.warn("Adding observation to a constant variable")

        override def addVariable(modelScore: Score, guideScore: Score): Unit =
          logger.warn("Adding dependant variable to a constant variable")
      }
  }

  trait Observation {

    def score: Score

    def complete(): Unit
  }

  trait Model[A] {

    def sample(): Variable[A]
  }

  trait Model1[X, A] {
    self =>

    def apply(in: X): Model[A] =
      () => self.sample(in)

    def apply(in: Variable[X]): Model[A] =
      () => self.sample(in)

    def sample(in: Variable[X]): Variable[A]
  }

  case class BBVIGuide[A](posterior: Distribution[A]) {

    var iter = 0

    // control variate
    // Since a constant delta between score_p and score_q has an expectation value of zero,
    // the average value can be subtracted in order to reduce the variance.
    var weight: Double = 0.0
    var offset: Double = 0.0

    // samples the guide (= the approximation to the posterior)
    // use BBVI (with Rao Blackwellization)
    def sample(prior: Distribution[A]): Variable[A] = new Variable[A] {

      private val sample = posterior.sample()
      private var modelScores = Set.empty[Score]
      private var guideScores = Set.empty[Score]

      override val get: A = {
        sample.get
      }

      override val modelScore: Buffer[Double] = {
        prior.observe(get).buffer
      }

      override val guideScore: Buffer[Double] = {
        posterior.observe(get).buffer
      }

      override def addObservation(score: Score): Unit = {
        modelScores += score
      }

      override def addVariable(modelScore: Score, guideScore: Score): Unit = {
        modelScores += modelScore
        guideScores += guideScore
      }

      // compute ELBO and backprop gradients
      override def complete(): Unit = {
        val logp: DValue[Double] = modelScore + modelScores.sum
        val logq: DValue[Double] = guideScore + guideScores.sum
        modelScore.dv(1.0)
        update(guideScore, logp, logq)
        modelScore.complete()
        guideScore.complete()
      }

    }

    /**
      * Backprop using BBVI - the guide (prior) score gradient is backpropagated
      * with as weight the Rao-Blackwellized delta between the model and guide
      * (full) score.  The average difference is used as the control variate, to reduce
      * variance of the gradient.
      */
    private def update(s: Score, logp: Score, logq: Score) = {
      iter += 1
      val rho = math.pow(iter, -0.5)

      val delta = logp.v - logq.v

      weight = (1.0 - rho) * weight + rho
      offset = (1.0 - rho) * offset + rho * delta
      val control = if (weight < 1e-12) {
        0.0
      }
      else {
        offset / weight
      }

//      println(s" BBVI delta: ${delta}, control: ${control}  ($iter)")

      s.dv(delta - control)
    }

  }

  case class Bernoulli(p: Variable[DValue[Double]]) extends Distribution[Boolean] {

    override def sample(): Sample[Boolean] = {
      val value =  Random.nextDouble() < p.get.v
//      println(s"Sample: $value (${p.get.v})")
      new Sample[Boolean] {

        override val get: Boolean =
          value

        override val score: Score =
          Bernoulli.this.observe(get)

        override def complete(): Unit = {}
      }
    }

    override def observe(value: Boolean): Score = {
      if (value) log(p.get) else log(-p.get + 1.0)
    }

  }

  object Bernoulli {

    def apply(p: Double): Bernoulli =
      Bernoulli(DValue.toConstant(p))
  }

  case class ReparamGuide[A](guide: DDistribution[A]) {

    def sample(model: DDistribution[A]): Variable[DValue[A]] = new Variable[DValue[A]] {

      private val s = guide.sample()

      override val get: DValue[A] =
        s.get

      override val modelScore: Buffer[Double] = {
        model.observe(get).buffer
      }

      override val guideScore: Buffer[Double] = {
        guide.reparam_score(get).buffer
      }

      override def addObservation(score: Score): Unit = {}

      override def addVariable(modelScore: Score, guideScore: Score): Unit = {}

      override def complete(): Unit = {
        modelScore.dv(1.0)
//        println("completing model score")
        modelScore.complete()
        guideScore.dv(-1.0)
//        println("completing guide score")
        guideScore.complete()
        s.complete()
      }
    }

  }

  case class Normal(mu: Variable[DValue[Double]], sigma: Variable[DValue[Double]]) extends DDistribution[Double] {
    dist =>

    override def sample(): Sample[DValue[Double]] = new Sample[DValue[Double]] {

      private val x = mu.get + sigma.get * Random.nextGaussian()
//      println(s"  x: ${x.v} (${mu.get.v}, ${sigma.get.v})")
//      new Exception().printStackTrace()

      override val get: Buffer[Double] = x.buffer

      override def score: Score = dist.observe(get)

      override def complete(): Unit = get.complete()
    }

    override def observe(x: DValue[Double]): Score = {
      -log(sigma.get) - pow((x - mu.get) / sigma.get, 2.0) / 2.0
    }

    override def reparam_score(x: DValue[Double]): Score = {
      -log(sigma.get.const) - pow((x - mu.get.const) / sigma.get.const, 2.0) / 2.0
    }
  }

  object Normal {

    def apply(mu: Double, sigma: Double): Normal =
      Normal(DValue.toConstant(mu), DValue.toConstant(sigma))
  }

}
