package scappla.guides

import scappla.{BayesNode, Buffer, Real, Score, Variable}
import scappla.distributions.DDistribution

case class ReparamGuide(posterior: DDistribution) {

  def sample(prior: DDistribution): Variable[Real] = {

    val sample = posterior.sample()
    val value: Real = sample.get

    val node = new BayesNode {
      override val modelScore: Buffer = {
        prior.observe(value).buffer
      }

      override val guideScore: Buffer = {
        posterior.reparam_score(value).buffer
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
        sample.complete()
      }
    }

    Variable(value, node)
  }

}
