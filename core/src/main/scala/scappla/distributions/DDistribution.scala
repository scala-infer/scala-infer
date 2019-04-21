package scappla.distributions

import scappla.{Buffered, Interpreter, Score, Value}

trait DDistribution[D] extends Distribution[Value[D]] {

  override def sample(interpreter: Interpreter): Buffered[D]

  /**
   * Score implementation that only back-propagates derivatives to the sampled variable,
   * not to the distribution parameters.
   */
  def reparam_score(interpreter: Interpreter, a: Value[D]): Score
}
