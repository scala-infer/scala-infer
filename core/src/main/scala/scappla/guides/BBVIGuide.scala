package scappla.guides

import scappla._
import scappla.distributions.Distribution
import scappla.optimization.Average

// control variate
// Since a constant delta between score_p and score_q has an expectation value of zero,
// the average value can be subtracted in order to reduce the variance.
case class BBVIGuide[A](posterior: Distribution[A], control: Value[Double] = Average.param(0.0)) extends Guide[A] {

  // samples the guide (= the approximation to the posterior)
  // use BBVI (with Rao Blackwellization)
  override def sample(interpreter: Interpreter, prior: Distribution[A]): Variable[A] = {

    val value: A = posterior.sample(interpreter)

    val node: BayesNode = new BayesNode {

      override val modelScore: Buffered[Double] = {
        prior.observe(interpreter, value).buffer
      }

      override val guideScore: Buffered[Double] = {
        posterior.observe(interpreter, value).buffer
      }

      private var logp: Score = modelScore
      private var logq: Score = guideScore

      override def addObservation(score: Score): Unit = {
        logp = DAdd(logp, score)
      }

      override def addVariable(modelScore: Score, guideScore: Score): Unit = {
        logp = DAdd(logp, modelScore)
        logq = DAdd(logq, guideScore)
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
  private def update(s: Score, logp: Score, logq: Score): Unit = {
    val delta = logp.v - logq.v

    val gradient = delta - control.v
    control.dv(gradient)

    s.dv(gradient)
  }

}
