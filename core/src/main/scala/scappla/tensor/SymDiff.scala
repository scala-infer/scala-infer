package scappla.tensor

import shapeless.Nat

trait ShapeDiffMappers[A, B, C] {

  def ab(a: A, b: B): C

  def bc(b: B, c: C): A

  def ca(c: C, a: A): B

  def shiftLeft: ShapeDiffMappers[B, C, A] = {
    val self = this
    new ShapeDiffMappers[B, C, A] {

      def ab(b: B, c: C): A = self.bc(b, c)

      def bc(c: C, a: A): B = self.ca(c, a)

      def ca(a: A, b: B): C = self.ab(a, b)
    }
  }

  def shiftRight: ShapeDiffMappers[C, A, B] = {
    val self = this
    new ShapeDiffMappers[C, A, B] {

      def ab(c: C, a: A): B = self.ca(c, a)

      def bc(a: A, b: B): C = self.ab(a, b)

      def ca(b: B, c: C): A = self.bc(b, c)
    }
  }

}

trait SymDiff[A, B] {
  type Out

  def mapper: ShapeDiffMappers[A, B, Out]

  def matchedIndices: List[(Int, Int)]

  def recoverLeft: SymDiff.Aux[B, Out, A]

  def recoverRight: SymDiff.Aux[Out, A, B]
}

case class ShapeSymDiffIndices(
    la: Int,
    lb: Int,
    mi: List[(Int, Int)]
)

class ShapeSymDiff[A, B, C](indices: ShapeSymDiffIndices, val mapper: ShapeDiffMappers[A, B, C]) extends SymDiff[A, B] {
  type Out = C

  override def matchedIndices: List[(Int, Int)] = indices.mi

  override def recoverLeft: SymDiff.Aux[B, C, A] = new ShapeSymDiffLeft(indices, mapper)

  override def recoverRight: SymDiff.Aux[C, A, B] = new ShapeSymDiffRight(indices, mapper)
}

class ShapeSymDiffLeft[A, B, C](indices: ShapeSymDiffIndices, origMapper: ShapeDiffMappers[A, B, C]) extends SymDiff[B, C] {
  type Out = A

  override val mapper: ShapeDiffMappers[B, C, A] = origMapper.shiftLeft

  // compute indices in B and C that match
  // == those indices in B that are not in A
  override def matchedIndices: List[(Int, Int)] = {
    val offset = indices.la - indices.mi.size
    val matchedB = indices.mi.map(_._2)
    (0 until indices.lb)
        .filter(ib => !matchedB.contains(ib))
        .zipWithIndex
        .map { case (bi, ci) => (bi, ci + offset) }
        .toList
  }

  override def recoverLeft: SymDiff.Aux[C, A, B] = new ShapeSymDiffRight(indices, origMapper)

  override def recoverRight: SymDiff.Aux[A, B, C] = new ShapeSymDiff(indices, origMapper)
}

class ShapeSymDiffRight[A, B, C](m: ShapeSymDiffIndices, origMapper: ShapeDiffMappers[A, B, C]) extends SymDiff[C, A] {
  type Out = B

  override def mapper: ShapeDiffMappers[C, A, B] = origMapper.shiftRight

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

  override def recoverLeft: SymDiff.Aux[A, B, C] = new ShapeSymDiff(m, origMapper)

  override def recoverRight: SymDiff.Aux[B, C, A] = new ShapeSymDiffLeft(m, origMapper)
}

object SymDiff extends LowPrioSymDiff {

  def apply[A <: Shape, B <: Shape](implicit o: SymDiff[A, B]): Aux[A, B, o.Out] = o

  type Aux[A, B, C] = SymDiff[A, B] {type Out = C}

  // A =:= Scalar
  implicit def symDiffScalarA[B <: Shape](implicit bl: Len[B]): Aux[Scalar, B, B] = {
    val mapper = new ShapeDiffMappers[Scalar, B, B] {
      override def ab(a: Scalar, b: B): B = b
      override def bc(b: B, c: B): Scalar = Scalar
      override def ca(c: B, a: Scalar): B = c
    }
    new ShapeSymDiff[Scalar, B, B](ShapeSymDiffIndices(0, bl.apply(), Nil), mapper)
  }

  // B =:= Scalar
  implicit def symDiffScalarB[A <: Shape](implicit al: Len[A]): Aux[A, Scalar, A] = {
    val mapper = new ShapeDiffMappers[A, Scalar, A] {
      override def ab(a: A, b: Scalar): A = a
      override def bc(b: Scalar, c: A): A = c
      override def ca(c: A, a: A): Scalar = Scalar
    }
    new ShapeSymDiff[A, Scalar, A](ShapeSymDiffIndices(al.apply(), 0, Nil), mapper)
  }

  // B =:= Scalar
  implicit def symDiffIdent[A <: Shape](implicit al: Len[A]): Aux[A, A, Scalar] = {
    val mapper = new ShapeDiffMappers[A, A, Scalar] {
      override def ab(a: A, b: A): Scalar = Scalar
      override def bc(b: A, c: Scalar): A = b
      override def ca(c: Scalar, a: A): A = a
    }
    val indices = ShapeSymDiffIndices(al(), al(), Range(0, al()).toList.map { i => (i, i) })
    new ShapeSymDiff[A, A, Scalar](indices, mapper)
  }

  // A.head ∈ B => A.head ∉ C
  implicit def symDiffHead[H <: Dim[_], B <: Shape, R <: Shape, N <: Nat, C <: Shape]
  (implicit
      idx: IndexOf.Aux[B, H, N],
      ia: InsertAt.Aux[R, N, H, B],
      s: SymDiff.Aux[Scalar, R, C],
      bl: Len[B]
  ): SymDiff.Aux[H, B, C] = {
    val matched = (0, idx.toInt) :: (s.matchedIndices map {
      case (i, j) => (i + 1, if (j >= idx.toInt) j + 1 else j)
    })
    val indices = ShapeSymDiffIndices(1, bl.apply(), matched)
    val mapper = new ShapeDiffMappers[H, B, C] {
      override def ab(a: H, b: B): C = s.mapper.ab(Scalar, ia.removeFrom(b))
      override def bc(b: B, c: C): H = idx.dimFrom(b)
      override def ca(c: C, a: H): B = {
        val rShape = s.mapper.ca(c, Scalar)
        ia.apply(rShape, a)
      }
    }
    new ShapeSymDiff[H, B, C](indices, mapper)
  }

}

trait LowPrioSymDiff {

  // A =:= Dim
  implicit def symDiffDimA[A <: Dim[_], B <: Shape]
  (implicit
      bl: Len[B],
      n: NotContains[B, A]
  ): SymDiff.Aux[A, B, A :#: B] = {
    val mapper = new ShapeDiffMappers[A, B, A :#: B] {
      override def ab(a: A, b: B): A :#: B = a :#: b
      override def bc(b: B, c: A :#: B): A = c.head
      override def ca(c: A :#: B, a: A): B = c.tail
    }
    val indices = ShapeSymDiffIndices(1, bl.apply(), Nil)
    new ShapeSymDiff[A, B, A :#: B](indices, mapper)
  }

  // A.head ∉ B => A.head ∈ C
  implicit def symDiffNoMatch[H <: Dim[_], T <: Shape, B <: Shape, C <: Shape]
  (implicit
      n: NotContains[B, H],
      s: SymDiff.Aux[T, B, C],
      bl: Len[B],
      tl: Len[T]
  ): SymDiff.Aux[H :#: T, B, H :#: C] = {
    val mapper = new ShapeDiffMappers[H :#: T, B, H:#: C] {
      override def ab(a: H :#: T, b: B): H :#: C = a.head :#: s.mapper.ab(a.tail, b)
      override def bc(b: B, c: H :#: C): H :#: T = c.head :#: s.mapper.bc(b, c.tail)
      override def ca(c: H :#: C, a: H :#: T): B = s.mapper.ca(c.tail, a.tail)
    }
    val matched = s.matchedIndices map { case (i, j) => (i + 1, j) }
    val indices = ShapeSymDiffIndices(tl.apply() + 1, bl.apply(), matched)
    new ShapeSymDiff[H :#: T, B, H :#: C](indices, mapper)
  }

  // A.head ∈ B => A.head ∉ C
  implicit def symDiffMatch[H <: Dim[_], T <: Shape, B <: Shape, R <: Shape, N <: Nat, C <: Shape]
  (implicit
      idx: IndexOf.Aux[B, H, N],
      ia: InsertAt.Aux[R, N, H, B],
      s: SymDiff.Aux[T, R, C],
      bl: Len[B],
      tl: Len[T]
  ): SymDiff.Aux[H :#: T, B, C] = {
    val mapper = new ShapeDiffMappers[H :#: T, B, C] {
      override def ab(a: H :#: T, b: B): C = s.mapper.ab(a.tail, ia.removeFrom(b))
      override def bc(b: B, c: C): H :#: T = ia.getElem(b) :#: s.mapper.bc(ia.removeFrom(b), c)
      override def ca(c: C, a: H :#: T): B = ia.apply(s.mapper.ca(c, a.tail), a.head)
    }
    val matched = (0, idx.toInt) :: (s.matchedIndices map {
      case (i, j) => (i + 1, if (j >= idx.toInt) j + 1 else j)
    })
    val indices = ShapeSymDiffIndices(tl.apply() + 1, bl.apply(), matched)
    new ShapeSymDiff[H :#: T, B, C](indices, mapper)
  }

}
