package scappla

import scappla.{Interpreter, Score}

trait Likelihood[A] {

  def observe(interpreter: Interpreter, a: A): Score
}
