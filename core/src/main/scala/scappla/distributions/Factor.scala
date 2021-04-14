package scappla.distributions

import scappla._

object Factor extends Likelihood[Score] {

  override def observe(interpreter: Interpreter, a: Score): Score = a
}
