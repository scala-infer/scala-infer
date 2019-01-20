package scappla.tensor

import shapeless.Nat

/*

trait SymDiff[A, B] {
  type Out
  def matchedIndices: List[(Int, Int)]
  def recoverLeft: SymDiff.Aux[B, Out, A]
  def recoverRight: SymDiff.Aux[Out, A, B]
}

case class ShapeSymDiffData(
    la: Int,
    lb: Int,
    mi: List[(Int, Int)]
)

class ShapeSymDiff[A, B, C](m: ShapeSymDiffData) extends SymDiff[A, B] {
  type Out = C

  override def matchedIndices: List[(Int, Int)] = m.mi

  override def recoverLeft: SymDiff.Aux[B, C, A] = new ShapeSymDiffLeft(m)

  override def recoverRight: SymDiff.Aux[C, A, B] = new ShapeSymDiffRight(m)
}

class ShapeSymDiffLeft[A, B, C](m: ShapeSymDiffData) extends SymDiff[B, C] {
  type Out = A

  // compute indices in B and C that match
  // == those indices in B that are not in A
  override def matchedIndices: List[(Int, Int)] = {
    val offset = m.la - m.mi.size
    val matchedB = m.mi.map(_._2)
    (0 until m.lb)
        .filter(ib => !matchedB.contains(ib))
        .zipWithIndex
        .map { case (bi, ci) => (bi, ci + offset) }
        .toList
  }

  override def recoverLeft: SymDiff.Aux[C, A, B] = new ShapeSymDiffRight(m)

  override def recoverRight: SymDiff.Aux[A, B, C] = new ShapeSymDiff(m)
}

class ShapeSymDiffRight[A, B, C](m: ShapeSymDiffData) extends SymDiff[C, A] {
  type Out = B

  // compute indices in A and C that match
  // == those indices in A that are not in B
  override def matchedIndices: List[(Int, Int)] = {
    val matchedA = m.mi.map(_._1)
    (0 until m.la)
        .filter(ia => !matchedA.contains(ia))
        .zipWithIndex
        .map(_.swap)
        .toList
  }

  override def recoverLeft: SymDiff.Aux[A, B, C] = new ShapeSymDiff(m)

  override def recoverRight: SymDiff.Aux[B, C, A] = new ShapeSymDiffLeft(m)
}

object SymDiff {

  def apply[A, B](implicit o: SymDiff[A, B]): Aux[A, B, o.Out] = o
  type Aux[A, B, C] = SymDiff[A, B] { type Out = C }

  // A =:= Scalar
  implicit def symDiffNil[B <: Shape](implicit bl: Len[B]): Aux[Scalar, B, B] =
    new ShapeSymDiff[Scalar, B, B](ShapeSymDiffData(0, bl.apply(), Nil))

  // A.head ∉ B => A.head ∈ C
  implicit def symDiffNoMatch[H, T <: Shape, B <: Shape, C <: Shape]
  (implicit n: NotContains[B, H], s: SymDiff.Aux[T, B, C], bl: Len[B], tl: Len[T]): Aux[H :: T, B, H :: C] = {
    val matched = s.matchedIndices map { case (i, j) => (i + 1, j) }
    new ShapeSymDiff[H :: T, B, H :: C](ShapeSymDiffData(tl.apply() + 1, bl.apply(), matched))
  }

  // A.head ∈ B => A.head ∉ C
  implicit def symDiffMatch[H, T <: Shape, B <: Shape, R <: Shape, N <: Nat, C <: Shape]
  (implicit idx: IndexOf.Aux[B, H, N], r: RemoveAt.Aux[B, N, R], s: SymDiff.Aux[T, R, C], bl: Len[B], tl: Len[T]): Aux[H :: T, B, C] = {
    val matched = (0, idx.toInt) :: (s.matchedIndices map { case (i, j) => (i + 1, if (j >= idx.toInt) j + 1 else j) })
    new ShapeSymDiff[H :: T, B, C](ShapeSymDiffData(tl.apply() + 1, bl.apply(), matched))
  }

}
*/