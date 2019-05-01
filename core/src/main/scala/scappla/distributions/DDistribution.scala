package scappla.distributions

import scappla.{Buffered, Interpreter, Score, Value}

trait DDistribution[D, S] extends Distribution[Value[D, S]] {

  override def sample(interpreter: Interpreter): Buffered[D, S]

  /**
   * Score implementation that only back-propagates derivatives to the sampled variable,
   * not to the distribution parameters.
   */
  def reparam_score(interpreter: Interpreter, a: Value[D, S]): Score
}
