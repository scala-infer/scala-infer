package scappla.distributions

import scappla.{Expr, Score}

trait DDistribution[D] extends Distribution[Expr[D]] {

  def reparam_score(a: Expr[D]): Score
}
