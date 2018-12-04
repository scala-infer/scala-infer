package scappla.guides

import scappla._
import scappla.distributions.Distribution
import scappla.Real._

case class BBVIGuide[A](posterior: Distribution[A]) extends Guide[A] {

  var iter = 0

  // control variate
  // Since a constant delta between score_p and score_q has an expectation value of zero,
  // the average value can be subtracted in order to reduce the variance.
  var weight: Double = 0.0
  var offset: Double = 0.0

  // samples the guide (= the approximation to the posterior)
  // use BBVI (with Rao Blackwellization)
  def sample(prior: Distribution[A]): Variable[A] = {

    val sample = posterior.sample()
    val value: A = sample.get

    val node: BayesNode = new BayesNode {

      override val modelScore: RealBuffer = {
        prior.observe(value).buffer
      }

      override val guideScore: RealBuffer = {
        posterior.observe(value).buffer
      }

      private var logp: Score = modelScore
      private var logq: Score = guideScore

      override def addObservation(score: Score): Unit = {
        logp += score
      }

      override def addVariable(modelScore: Score, guideScore: Score): Unit = {
        logp += modelScore
        logq += guideScore
      }

      // compute ELBO and backprop gradients
      override def complete(): Unit = {
        // backprop gradients to decoder
        modelScore.dv(1.0)

        // backprop gradients to encoder
        update(guideScore, logp, logq)

        // evaluate optimizer
        modelScore.complete()
        guideScore.complete()
      }
    }

    Variable(value, node)
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
