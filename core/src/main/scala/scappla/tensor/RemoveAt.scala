package scappla.tensor

import shapeless._

import scala.annotation.implicitNotFound

trait RemoveAt[L <: Shape, I <: Nat] extends DepFn1[L] {
  type Out
  def apply(in: L): Out
}

object RemoveAt extends LowPriorityRemoveAtImplicits {

  @implicitNotFound("Could not find RemoveAt implicit to apply")
  def apply[L <: Shape, I <: Nat](implicit o: RemoveAt[L, I]): Aux[L, I, o.Out] = o
  type Aux[L <: Shape, I <: Nat, Out0] = RemoveAt[L, I] { type Out = Out0 }

  implicit def removeAtShapeCase1[T <: Shape, H <: Dim[_]]: Aux[H :#: T, _0, T] =
    new RemoveAt[H :#: T, _0] {
      type Out = T
      def apply(t: H :#: T): T = t.tail
    }

  implicit def removeAtShapeCase2[T <: Dim[_], H <: Dim[_]]
  (implicit r: RemoveAt.Aux[T, _0, Scalar]): Aux[H :#: T, Succ[_0], H] =
    new RemoveAt[H :#: T, Succ[_0]] {
      type Out = H
      def apply(t: H :#: T): H = t.head
    }

}

trait LowPriorityRemoveAtImplicits {

  implicit def removeAtShapeCaseN[T <: Shape, H <: Dim[_], R <: Shape, P <: Nat]
  (implicit r: RemoveAt.Aux[T, P, R]): RemoveAt.Aux[H :#: T, Succ[P], H :#: R] =
    new RemoveAt[H :#: T, Succ[P]] {
      type Out = H :#: R
      def apply(t: H :#: T): H :#: R = t.head :#: r(t.tail)
    }

  implicit def removeAtShapeCase0[H <: Dim[_]]: RemoveAt.Aux[H, _0, Scalar] =
    new RemoveAt[H, _0] {
      type Out = Scalar
      def apply(t: H): Scalar = Scalar
    }

}
