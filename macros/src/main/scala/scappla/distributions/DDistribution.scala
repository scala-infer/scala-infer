package scappla.distributions

import scappla.{Real, Score}

trait DDistribution extends Distribution[Real] {

    def reparam_score(a: Real): Score
  }
