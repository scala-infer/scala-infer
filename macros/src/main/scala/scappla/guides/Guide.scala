package scappla.guides

import scappla.Variable
import scappla.distributions.Distribution

trait Guide[A] {
  def sample(prior: Distribution[A]): Variable[A]
}
