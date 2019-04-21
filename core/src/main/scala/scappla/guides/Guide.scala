package scappla.guides

import scappla.{Interpreter, Variable}
import scappla.distributions.Distribution

trait Guide[A] {

  def sample(interpreter: Interpreter, prior: Distribution[A]): Variable[A]
}
