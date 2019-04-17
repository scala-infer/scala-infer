package scappla.distributions

import scappla.{Buffered, Value, Score}

trait DDistribution[D] extends Distribution[Value[D]] {

  override def sample(): Buffered[D]

  /**
   * Score implementation that only back-propagates derivatives to the sampled variable,
   * not to the distribution parameters.
   */
  def reparam_score(a: Value[D]): Score
}
