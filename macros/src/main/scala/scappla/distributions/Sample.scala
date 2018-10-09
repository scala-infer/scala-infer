package scappla.distributions

import scappla.Score

trait Sample[A] {

    def get: A

    def score: Score

    def complete(): Unit
  }
