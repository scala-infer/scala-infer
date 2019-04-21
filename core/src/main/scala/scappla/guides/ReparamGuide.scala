package scappla.guides

import scappla._
import scappla.distributions.{DDistribution, Distribution}

case class ReparamGuide[D](posterior: DDistribution[D]) extends Guide[Value[D]] {

  override def sample(interpreter: Interpreter, prior: Distribution[Value[D]]): Variable[Value[D]] = {

    val value = posterior.sample(interpreter)

    val node = new BayesNode {
      override val modelScore: Buffered[Double] = {
        prior.observe(interpreter, value).buffer
      }

      override val guideScore: Buffered[Double] = {
        posterior.reparam_score(interpreter, value).buffer
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
