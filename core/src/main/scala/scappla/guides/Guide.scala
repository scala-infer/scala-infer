package scappla.guides

import scappla.{Interpreter, Variable}
import scappla.Likelihood

trait Guide[A] {

  def sample(interpreter: Interpreter, prior: Likelihood[A]): Variable[A]
}
