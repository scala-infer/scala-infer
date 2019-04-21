package scappla.distributions

import scappla.{Interpreter, Sampleable, Score}

trait Distribution[A] extends Sampleable[A] {

  def observe(interpreter: Interpreter, a: A): Score
}
