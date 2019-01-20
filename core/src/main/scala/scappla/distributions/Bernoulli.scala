package scappla.distributions

import scappla.Functions.log
import scappla.{Real, Score}

import scala.util.Random

case class Bernoulli(p: Real) extends Distribution[Boolean] {

  import Real._

  override def sample(): Boolean = {
    Random.nextDouble() < p.v
  }

  override def observe(value: Boolean): Score = {
    if (value) log(p) else log(Real(1.0) - p)
  }

}
