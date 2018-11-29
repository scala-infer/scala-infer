package scappla.tensor

import scappla.Functions.{exp, log}
import scappla.{Functions, Real}
import shapeless.Nat

sealed trait Shape {

  def size: Int

  def sizes: List[Int]
}

object Shape {
  implicit def ops[S <: Shape](shape: S) = new ShapeOps[S](shape)
}

class ShapeOps[S <: Shape](shape: S) {
  def :#:[H <: Dim[_]](h : H) : H :#: S = scappla.tensor.:#:(h, shape)
}

trait Dim[Self <: Dim[_]] extends Shape {
  self: Self =>

  def size: Int

  final override def sizes: List[Int] = List(size)

  def :#:[H <: Dim[_]](head: H) = scappla.tensor.:#:[H, Self](head, this)
}

final case class :#:[H <: Dim[_], +T <: Shape](head: H, tail: T) extends Shape {

  def size = head.size * tail.size

  def sizes = head.size :: tail.sizes
}

sealed trait Scalar extends Shape {

  val size = 1

  val sizes = List.empty
}

object Scalar extends Scalar


sealed trait Tensor[S <: Shape] {

  def shape: S

  def plus(other: Tensor[S]): Tensor[S] = {
    TPlus(this, other)
  }

  def minus(other: Tensor[S]): Tensor[S] = {
    TMinus(this, other)
  }

  def times(other: Tensor[S]): Tensor[S] = {
    TTimes(this, other)
  }

  def div(other: Tensor[S]): Tensor[S] = {
    TDiv(this, other)
  }

  def negate: Tensor[S] = {
    TNeg(this)
  }

}

trait TensorInterpreter {

  def interpret(tensor: Tensor[Scalar], resolver: TParam[_] => Array[Float]): Real
}

case class TParam[S <: Shape](shape: S, backward: Array[Float] => Unit) extends Tensor[S]

case class TNeg[S <: Shape](orig: Tensor[S]) extends Tensor[S] {

  override val shape : S = orig.shape
}

case class TPlus[S <: Shape](left: Tensor[S], right: Tensor[S]) extends Tensor[S] {
  assert(left.shape == right.shape)

  override val shape: S = left.shape
}

case class TMinus[S <: Shape](left: Tensor[S], right: Tensor[S]) extends Tensor[S] {
  assert(left.shape == right.shape)

  override val shape: S = left.shape
}

case class TTimes[S <: Shape](left: Tensor[S], right: Tensor[S]) extends Tensor[S] {
  assert(left.shape == right.shape)

  override val shape: S = left.shape
}

case class TDiv[S <: Shape](numer: Tensor[S], denom: Tensor[S]) extends Tensor[S] {
  assert(numer.shape == denom.shape)

  override val shape: S = numer.shape
}

case class TLog[S <: Shape](upstream: Tensor[S]) extends Tensor[S] {

  override val shape: S = upstream.shape
}

case class TExp[S <: Shape](upstream: Tensor[S]) extends Tensor[S] {

  override val shape: S = upstream.shape
}

case class TSum[R <: Shape, S <: Shape](shape: R, index: Int, upstream: Tensor[S]) extends Tensor[R]

object Tensor {

  def sum[S <: Shape,D <: Dim[_], I <: Nat, R <: Shape](
      tensor: Tensor[S]
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): Tensor[R] = {
    TSum[R, S](removeAt.apply(tensor.shape), indexOf.toInt, tensor)
  }

  // NUMERIC

  implicit def numericForTensor[S <: Shape]: Fractional[Tensor[S]] = new Fractional[Tensor[S]] {

    override def plus(x: Tensor[S], y: Tensor[S]): Tensor[S] = x.plus(y)

    override def minus(x: Tensor[S], y: Tensor[S]): Tensor[S] = x.minus(y)

    override def times(x: Tensor[S], y: Tensor[S]): Tensor[S] = x.times(y)

    override def div(x: Tensor[S], y: Tensor[S]): Tensor[S] = x.div(y)

    override def negate(x: Tensor[S]): Tensor[S] = x.negate

    override def fromInt(x: Int): Tensor[S] = ???

    override def toInt(x: Tensor[S]): Int = ???

    override def toLong(x: Tensor[S]): Long = ???

    override def toFloat(x: Tensor[S]): Float = ???

    override def toDouble(x: Tensor[S]): Double = ???

    override def compare(x: Tensor[S], y: Tensor[S]): Int = ???

  }

  // FUNCTIONS

  implicit def logTensor[S <: Shape]: log.Apply[Tensor[S], Tensor[S]] = new Functions.log.Apply[Tensor[S], Tensor[S]] {

    def apply(in: Tensor[S]): Tensor[S] = TLog(in)
  }

  implicit def expTensor[S <: Shape]: exp.Apply[Tensor[S], Tensor[S]] = new Functions.exp.Apply[Tensor[S], Tensor[S]] {

    def apply(in: Tensor[S]): Tensor[S] = TExp(in)
  }
}
