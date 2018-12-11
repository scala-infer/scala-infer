package scappla.guides

import scappla._
import scappla.distributions.{DDistribution, Distribution}

case class ReparamGuide[D](posterior: DDistribution[D]) extends Guide[Expr[D]] {

  def sample(prior: Distribution[Expr[D]]): Variable[Expr[D]] = {

    val sample = posterior.sample()
    val value: Expr[D] = sample.get

    val node = new BayesNode {
      override val modelScore: Buffered[Double] = {
        prior.observe(value).buffer
      }

      override val guideScore: Buffered[Double] = {
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
