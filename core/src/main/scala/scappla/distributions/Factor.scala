package scappla.distributions

import scappla._

object Factor extends Distribution[Score] {

  override def sample(interpreter: Interpreter): Score = ???

  override def observe(interpreter: Interpreter, a: Score): Score = a
}
