package scappla.distributions

import scappla.{Buffered, Expr, Score}

trait DDistribution[D] extends Distribution[Expr[D]] {

  override def sample(): Buffered[D]

  def reparam_score(a: Expr[D]): Score
}
