package scappla.tensor

import shapeless._

trait Len[L <: Shape] {
  type Out <: Nat
  def apply(): Int
}

object Len {

  def apply[L <: Shape](implicit l: Len[L]): Aux[L, l.Out] = l
  type Aux[L <: Shape, N <: Nat] = Len[L] { type Out = N }

  implicit def lenShapeCase0: Len.Aux[Scalar, _0] = new Len[Scalar] {
    type Out = _0
    def apply() = 0
  }

  implicit def lenShapeCase1[D <: Dim[_]]: Len.Aux[D, Succ[_0]] = new Len[D] {
    type Out = Succ[_0]
    def apply() = 1
  }

  implicit def lenShapeCaseN[H <: Dim[_], T <: Shape, P <: Nat](implicit t: Len.Aux[T, P]): Len.Aux[H :#: T, Succ[P]] = new Len[H :#: T] {
    type Out = Succ[P]
    override def apply() = t() + 1
  }

}