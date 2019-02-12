package scappla.tensor

import shapeless._

import scala.annotation.implicitNotFound

trait InsertAt[L <: Shape, I <: Nat, H <: Dim[_]] {
  type Out

  def apply(in: L, h: H): Out

  def removeFrom(out: Out): L

  def getElem(out: Out): H
}

object InsertAt extends LowPriorityInsertAtImplicits {

  @implicitNotFound("Could not find InsertAt implicit to apply")
  def apply[L <: Shape, I <: Nat, H <: Dim[_]](implicit o: InsertAt[L, I, H]): Aux[L, I, H, o.Out] = o

  type Aux[L <: Shape, I <: Nat, H <: Dim[_], Out0] = InsertAt[L, I, H] {type Out = Out0}

  implicit def headToScalarIsDim[H <: Dim[_]]: Aux[Scalar, _0, H, H] =
    new InsertAt[Scalar, _0, H] {

      type Out = H

      def apply(t: Scalar, h: H): H = h

      def removeFrom(out: H): Scalar = Scalar

      override def getElem(out: H): H = out
    }

  implicit def insertAtShapeScalarTail[H <: Dim[_], X <: Dim[_]]
  (implicit
      r: InsertAt.Aux[Scalar, _0, X, X]
  ): InsertAt.Aux[H, Succ[_0], X, H :#: X] =
    new InsertAt[H, Succ[_0], X] {

      type Out = H :#: X

      def apply(t: H, h: X): H :#: X = :#:(t, h)

      def removeFrom(out: H :#: X): H = {
        out.head
      }

      override def getElem(out: H :#: X): X = out.tail
    }

}

trait LowPriorityInsertAtImplicits {

  implicit def insertAtShapeCase1[T <: Shape, H <: Dim[_]]: InsertAt.Aux[T, _0, H, H :#: T] =
    new InsertAt[T, _0, H] {

      type Out = H :#: T

      def apply(t: T, h: H): H :#: T = h :#: t

      def removeFrom(out: H :#: T): T = out.tail

      override def getElem(out: H :#: T): H = out.head
    }

  implicit def insertAtShapeCaseN[T <: Shape, H <: Dim[_], X <: Dim[_], R <: Shape, P <: Nat]
  (implicit
      r: InsertAt.Aux[T, P, X, R]
  ): InsertAt.Aux[H :#: T, Succ[P], X, H :#: R] =
    new InsertAt[H :#: T, Succ[P], X] {

      type Out = H :#: R

      def apply(t: H :#: T, h: X): H :#: R = t.head :#: r.apply(t.tail, h)

      def removeFrom(out: :#:[H, R]): H :#: T = {
        out.head :#: r.removeFrom(out.tail)
      }

      override def getElem(out: H :#: R): X = r.getElem(out.tail)
    }

}
