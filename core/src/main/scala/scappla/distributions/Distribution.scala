package scappla.distributions

import scappla.{Sampleable, Score}

trait Distribution[A] extends Sampleable[A] {

  def observe(a: A): Score
}
