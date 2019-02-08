package scappla.tensor

import shapeless._

trait NotContains[L, X]

object NotContains {

  def apply[L <: Shape, X <: Dim[_]](implicit n: NotContains[L, X]) = n

  implicit def scalarContainsNothing[X <: Dim[_]]: NotContains[Scalar, X] = {
    new NotContains[Scalar, X] {}
  }

  implicit def neqDimsDontContain[H <: Dim[_], X <: Dim[_]]
  (implicit n: H =:!= X): NotContains[H, X] = {
    new NotContains[H, X] {}
  }

  implicit def higherOrderShape[H <: Dim[_], T <: Shape, X <: Dim[_]]
  (implicit t: NotContains[T, X], n: H =:!= X): NotContains[H :#: T, X] = {
    new NotContains[H :#: T, X] {}
  }

}
