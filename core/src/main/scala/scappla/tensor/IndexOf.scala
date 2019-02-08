package scappla.tensor

import shapeless._

trait IndexOf[L <: Shape, X <: Dim[_]] extends DepFn0 {
  type Out <: Nat
  def toInt: Int
}

object IndexOf {

  def apply[L <: Shape, X <: Dim[_]](implicit o: IndexOf[L, X]): Aux[L, X, o.Out] = o
  type Aux[L <: Shape, X <: Dim[_], Out0 <: Nat] = IndexOf[L, X] { type Out = Out0 }

  implicit def indexOfDimAtEnd[X <: Dim[_]]: Aux[X, X, _0] =
    new IndexOf[X, X] {
      type Out = _0
      def apply() = Nat._0
      def toInt = 0
    }

  implicit def indexOfDimAtHead[T <: Shape, X <: Dim[_]]: Aux[X :#: T, X, _0] =
    new IndexOf[X :#: T, X] {
      type Out = _0
      def apply() = Nat._0
      def toInt = 0
    }

  implicit def indexOfDimInList[T <: Shape, H <: Dim[_], X <: Dim[_], I <: Nat]
  (implicit p: IndexOf.Aux[T, X, I]): Aux[H :#: T, X, Succ[I]] =
    new IndexOf[H :#: T, X] {
      type Out = Succ[I]
      def apply() = Succ[I]()
      def toInt = p.toInt + 1
    }

}