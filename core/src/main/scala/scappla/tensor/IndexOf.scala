package scappla.tensor

import shapeless._

trait IndexOf[L <: Shape, X <: Dim[_]] {
  type Out <: Nat

  def dimFrom(shape: L): X

  def toInt: Int
}

object IndexOf {

  def apply[L <: Shape, X <: Dim[_]](implicit o: IndexOf[L, X]): Aux[L, X, o.Out] = o

  type Aux[L <: Shape, X <: Dim[_], Out0 <: Nat] = IndexOf[L, X] {type Out = Out0}

  implicit def indexOfDimAtEnd[X <: Dim[_]]: Aux[X, X, _0] =
    new IndexOf[X, X] {

      type Out = _0

      override def dimFrom(dim: X): X = dim

      override def toInt: Int = 0
    }

  implicit def indexOfDimAtHead[T <: Shape, X <: Dim[_]]: Aux[X :#: T, X, _0] =
    new IndexOf[X :#: T, X] {

      type Out = _0

      override def dimFrom(shape: X :#: T): X = shape.head

      override def toInt: Int = 0
    }

  implicit def indexOfDimInList[T <: Shape, H <: Dim[_], X <: Dim[_], I <: Nat]
  (implicit p: IndexOf.Aux[T, X, I]): Aux[H :#: T, X, Succ[I]] =
    new IndexOf[H :#: T, X] {

      type Out = Succ[I]

      override def dimFrom(shape: H :#: T): X = p.dimFrom(shape.tail)

      override def toInt: Int = p.toInt + 1
    }

}