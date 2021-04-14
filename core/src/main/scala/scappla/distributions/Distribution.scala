package scappla.distributions

import scappla.{Interpreter, Sampleable, Score}
import scappla.Likelihood

trait Distribution[A] extends Sampleable[A] with Likelihood[A] {
}
