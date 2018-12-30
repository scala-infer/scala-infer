package scappla.distributions

import scappla.Score

trait Distribution[A] {

  def sample(): A

  def observe(a: A): Score
}
