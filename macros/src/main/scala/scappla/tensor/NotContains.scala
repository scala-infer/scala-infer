package scappla.tensor

import shapeless._

/*
trait NotContains[L, X]

object NotContains {

  def apply[L, X](implicit n: NotContains[L, X]) = n

  implicit def notContainsHListCase0[X]: NotContains[Scalar, X]
  = new NotContains[Scalar, X] {}

  implicit def notContainsHListCaseN[H, T <: Shape, X]
  (implicit t: NotContains[T, X], n: H =:!= X): NotContains[H :: T, X]
  = new NotContains[H :: T, X] {}

}
*/