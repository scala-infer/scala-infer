package scappla.guides

import scappla._
import scappla.distributions.{DDistribution, Distribution}

case class ReparamGuide(posterior: DDistribution) extends Guide[Real] {

  def sample(prior: Distribution[Real]): Variable[Real] = {

    val sample = posterior.sample()
    val value: Real = sample.get

    val node = new BayesNode {
      override val modelScore: RealBuffer = {
        prior.observe(value).buffer
      }

      override val guideScore: RealBuffer = {
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
