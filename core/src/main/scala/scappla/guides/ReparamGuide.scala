package scappla.guides

import scappla._
import scappla.distributions.{DDistribution, Distribution}

case class ReparamGuide[D, S](posterior: DDistribution[D, S]) extends Guide[Value[D, S]] {

  override def sample(interpreter: Interpreter, prior: Distribution[Value[D, S]]): Variable[Value[D, S]] = {

    val value = posterior.sample(interpreter)

    val node = new BayesNode {

      override val modelScore: Buffered[Double, Unit] = {
        prior.observe(interpreter, value)
      }

      override val guideScore: Buffered[Double, Unit] = {
        posterior.reparam_score(interpreter, value)
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
        value.complete()
      }
    }

    Variable(value, node)
  }

}
