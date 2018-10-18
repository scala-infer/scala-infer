import com.typesafe.scalalogging.LazyLogging
import scappla.autodiff.AutoDiff
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

  /**
    *  Higher order functions.
    *  Iterating over collections and the like is (for now?) restricted to a limited
    *  set of explicitly functions.
    */
  def foreach[A](fn: A => Unit, m: List[A]): Unit = {}

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

  def foreachImpl[A](fn: Variable[A] => Variable[Unit], m: Variable[List[A]]): Variable[Unit] = {
    val Variable(values, node) = m
    val result = values.map { v =>
      fn(Variable(v, node))
    }
    val nodes = result.map(_.node)
    Variable(Unit, new BayesNode {

      override val modelScore: Score =
        nodes.foldLeft(Real(0.0)) { case (s, b) =>
          DAdd(s, b.modelScore)
        }

      override val guideScore: Score =
        nodes.foldLeft(Real(0.0)) { case (s, b) =>
          DAdd(s, b.guideScore)
        }

      override def addObservation(score: Score): Unit = ???

      override def addVariable(modelScore: Score, guideScore: Score): Unit = ???

      override def complete(): Unit = {
        for { node <- nodes } {
          node.complete()
        }
      }
    })
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
      logger.warn("Adding observation to a constant variable")

    override def addVariable(modelScore: Score, guideScore: Score): Unit =
      logger.warn("Adding dependant variable to a constant variable")
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

  def liftFunction[A, B](
      vf: Variable[A => B],
      fv: Variable[A]
  ): Variable[B] = {
    val Variable(f, lnode) = vf
    Variable(f(fv.get), new Dependencies(Seq(lnode, fv.node)))
  }

  val applied = liftFunction[Int, Double](
    Variable((i: Int) => 2.0 * i, ConstantNode),
    Variable(2, ConstantNode)
  )

  def liftHOMethod[A, B, C](
      vf: Variable[(A => B) => C],
      fv: Variable[A] => Variable[B]
  ): Variable[C] = {
    val Variable(f, lnode) = vf
    var nodes: List[BayesNode] = lnode :: Nil
    val lb = f { a =>
      val varA = Variable(a, lnode)
      val Variable(b, bnode) = fv(varA)
      nodes = bnode :: nodes
      b
    }
    Variable(lb, new Dependencies(nodes))
  }

  val testfv : Variable[Int] => Variable[Double] = vi => {
    val Variable(i, node) = vi
    Variable(2.0 * i, node)
  }

  val liftedMethod = liftHOMethod[Int, Double, List[Double]](
    Variable(List.empty[Int].map[Double, List[Double]], ConstantNode),
    testfv
  )

  def liftFn[A, B](
      fn: A => B
  ): Variable[A] => Variable[B] =
    (varA: Variable[A]) => {
      val Variable(a, adep) = varA
      Variable(fn(a), adep)
  }

}
