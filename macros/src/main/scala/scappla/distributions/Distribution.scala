package scappla.distributions

import scappla.Score

trait Distribution[A] {

    def sample(): Sample[A]

    def observe(a: A): Score
  }
