import com.typesafe.scalalogging.LazyLogging
import scappla.distributions.Distribution

import scala.language.experimental.macros

package object scappla {

  type Score = Real

  // API

  trait Model[A] {

    def sample(): Variable[A]
  }

  /**
   * The outermost "sample" is public - it drives the inference.  Each sample that's obtained
   * is used to further optimize the inferred distribution.  This can be by means of Monte Carlo,
   * or Variational Inference.
   */
  def sample[X](model: Model[X]): X = {
    val Variable(value, node) = model.sample()

    // prepare for next sample - update approximation
    node.complete()
    value
  }

  /**
    * Magic - transform "regular" scala code that produces a value into a model.
    */
  def infer[X](fn: X): scappla.Model[X] =
  macro Macros.infer[X]

  /**
    * Sampling in an infer block.
    * This implementation will be overridden by a macro.
    */
  def sample[X](prior: Distribution[X], posterior: Distribution[X]): X =
    posterior.sample().get

  /**
    * Register an observation in an infer block.
    */
  def observe[A](distribution: Distribution[A], value: A): Unit = {}

  // IMPLEMENTATION

  trait Observation extends Completeable {

    def score: Score
  }

  def sampleImpl[A](distribution: Model[A]): Variable[A] = {
    distribution.sample()
  }

  def observeImpl[A](distribution: Distribution[A], value: A): Observation =
    new Observation {

      val score: Buffer =
        distribution.observe(value).buffer

      override def complete(): Unit = {
        score.dv(1.0)
//        println(s"completing observation score ${distribution}")
        score.complete()
      }
    }

  /**
    * When the execution trace is torn down, each object is "completed" in reverse (topological)
    * order.  I.e. all objects that are dependent on the current object have been completed.
    * The complete operation is the last operation that will be invoked on the object.
    */
  trait Completeable {

    /**
      * When the value of the variable is retrieved and it can no longer be
      * exposed to any new dependencies, it should be completed.  In the case of the
      * outer "sample", this means updating the inferred distributions.
      */
    def complete(): Unit
  }

  trait BayesNode extends Completeable {

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
  }

  object ConstantNode extends BayesNode with LazyLogging {

    override def modelScore: Score = 0.0

    override def guideScore: Score = 0.0

    override def complete(): Unit = {}

    override def addObservation(score: Score): Unit =
      logger.debug("Adding observation to a constant variable")

    override def addVariable(modelScore: Score, guideScore: Score): Unit =
      logger.debug("Adding dependant variable to a constant variable")
  }

  class Dependencies(upstream: Seq[BayesNode]) extends BayesNode {

    val modelScore = 0.0

    val guideScore = 0.0

    override def addObservation(score: Score): Unit = {
      for { v <- upstream } v.addObservation(score)
    }

    override def addVariable(modelScore: Score, guideScore: Score): Unit = {
      for { v <- upstream } {
        v.addVariable(modelScore, guideScore)
      }
    }

    override def complete(): Unit = {}
  }

  // TODO: change inheritance to composition?
  case class Variable[A](get: A, node: BayesNode)

  object Variable extends LazyLogging {

    implicit def toConstant[A](value: A): Variable[A] = Variable[A](value, ConstantNode)
  }
}
