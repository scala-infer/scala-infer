package scappla.distributions

import scappla.{Buffered, Expr, Score}

trait DDistribution[D] extends Distribution[Expr[D]] {

  override def sample(): Buffered[D]

  /**
   * Score implementation that only back-propagates derivatives to the sampled variable,
   * not to the distribution parameters.
   */
  def reparam_score(a: Expr[D]): Score
}
