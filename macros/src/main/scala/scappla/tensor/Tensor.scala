package scappla.tensor

import scappla.Functions
import scappla.Functions.{exp, log}
import shapeless.Nat

sealed trait Shape {

  def size: Int

  def sizes: List[Int]
}

object Shape {
  implicit def ops[S <: Shape](shape: S) = new ShapeOps[S](shape)
}

class ShapeOps[S <: Shape](shape: S) {
  def :#:[H <: Dim[_]](h: H): H :#: S = scappla.tensor.:#:(h, shape)
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

trait DataOps[D] {

  // (de)constructing values

  def zeros(dims: Int*): D

  def set(a: Array[Float], dims: Int*): D

  def get(a: D): Array[Float]

  // element-wise operations

  def plus(a: D, b: D): D

  def minus(a: D, b: D): D

  def times(a: D, b: D): D

  def div(a: D, b: D): D

  def negate(a: D): D

  def log(a: D): D

  def exp(a: D): D

  // shape-affecting operations

  def sum(a: D, dim: Int): D

  def broadcast(a: D, dimIndex: Int, dimSize: Int): D
}

sealed trait Tensor[S <: Shape, D] {

  def dataOps: DataOps[D]

  def shape: S

  def forward: D

  def backward(gradient: D): Unit

  def plus(other: Tensor[S, D]): Tensor[S, D] = {
    TPlus(this, other)
  }

  def minus(other: Tensor[S, D]): Tensor[S, D] = {
    TMinus(this, other)
  }

  def times(other: Tensor[S, D]): Tensor[S, D] = {
    TTimes(this, other)
  }

  def div(other: Tensor[S, D]): Tensor[S, D] = {
    TDiv(this, other)
  }

  def negate: Tensor[S, D] = {
    TNeg(this)
  }
}

trait TensorInterpreter[D] {

  // create parameter with initial values and update function
  // update function receives gradient, returns new values
  def param[S <: Shape](values: D, update: (D, D) => D): TParam[S, D]

  def forward(tensor: Tensor[Scalar, D]): Float

  def backward(tensor: Tensor[Scalar, D], gradient: Float): Unit
}

case class TParam[S <: Shape, D](
    shape: S,
    dataOps: DataOps[D],
    values: () => D,
    update: D => Unit
) extends Tensor[S, D] {

  override def forward: D =
    values()

  override def backward(gradient: D): Unit =
    update(gradient)
}

case class TConst[S <: Shape, D](
    shape: S, dataOps: DataOps[D], values: Array[Float]
) extends Tensor[S, D] {

  override def forward: D =
    dataOps.set(values, shape.sizes.toArray: _*)

  override def backward(gradient: D): Unit = {}
}

case class TNeg[S <: Shape, D](orig: Tensor[S, D]) extends Tensor[S, D] {

  override val shape: S =
    orig.shape

  override val dataOps: DataOps[D] =
    orig.dataOps

  override def forward: D =
    dataOps.negate(orig.forward)

  override def backward(gradient: D): Unit =
    orig.backward(dataOps.negate(gradient))
}

case class TPlus[S <: Shape, D](left: Tensor[S, D], right: Tensor[S, D]) extends Tensor[S, D] {
  assert(left.shape == right.shape)

  override val shape: S =
    left.shape

  override val dataOps: DataOps[D] =
    left.dataOps

  override def forward: D =
    dataOps.plus(left.forward, right.forward)

  override def backward(gradient: D): Unit = {
    left.backward(gradient)
    right.backward(gradient)
  }
}

case class TMinus[S <: Shape, D](left: Tensor[S, D], right: Tensor[S, D]) extends Tensor[S, D] {
  assert(left.shape == right.shape)

  override val shape: S =
    left.shape

  override val dataOps: DataOps[D] =
    left.dataOps

  override def forward: D =
    dataOps.minus(left.forward, right.forward)

  override def backward(gradient: D): Unit = {
    left.backward(gradient)
    right.backward(dataOps.negate(gradient))
  }
}

case class TTimes[S <: Shape, D](left: Tensor[S, D], right: Tensor[S, D]) extends Tensor[S, D] {
  assert(left.shape == right.shape)

  override val shape: S =
    left.shape

  override val dataOps: DataOps[D] =
    left.dataOps

  override def forward: D =
    dataOps.times(left.forward, right.forward)

  override def backward(gradient: D): Unit = {
    left.backward(dataOps.times(gradient, right.forward))
    right.backward(dataOps.times(gradient, left.forward))
  }
}

case class TDiv[S <: Shape, D](numer: Tensor[S, D], denom: Tensor[S, D]) extends Tensor[S, D] {
  assert(numer.shape == denom.shape)

  override val shape: S =
    numer.shape

  override val dataOps: DataOps[D] =
    numer.dataOps

  override def forward: D =
    dataOps.div(numer.forward, denom.forward)

  override def backward(gradient: D): Unit = {
    numer.backward(dataOps.div(gradient, denom.forward))
    denom.backward(dataOps.div(
      dataOps.times(gradient, numer.forward),
      dataOps.times(denom.forward, denom.forward)
    ))
  }
}

case class TLog[S <: Shape, D](upstream: Tensor[S, D]) extends Tensor[S, D] {

  override val shape: S = upstream.shape

  override val dataOps: DataOps[D] = upstream.dataOps

  override def forward: D =
    dataOps.log(upstream.forward)

  override def backward(gradient: D): Unit =
    upstream.backward(dataOps.div(gradient, upstream.forward))
}

case class TExp[S <: Shape, D](upstream: Tensor[S, D]) extends Tensor[S, D] {

  override val shape: S =
    upstream.shape

  override val dataOps: DataOps[D] =
    upstream.dataOps

  override def forward: D =
    dataOps.exp(upstream.forward)

  override def backward(gradient: D): Unit =
    upstream.backward(dataOps.times(gradient, upstream.forward))
}

case class TSum[R <: Shape, S <: Shape, D](
    shape: R, index: Int, upstream: Tensor[S, D]
) extends Tensor[R, D] {

  override val dataOps: DataOps[D] =
    upstream.dataOps

  override def forward: D =
    dataOps.sum(upstream.forward, index)

  override def backward(gradient: D): Unit = {
    upstream.backward(dataOps.broadcast(gradient, index, upstream.shape.sizes(index)))
  }
}

object Tensor {

  def sum[S <: Shape, D <: Dim[_], I <: Nat, R <: Shape, X](
      tensor: Tensor[S, X]
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): Tensor[R, X] = {
    TSum[R, S, X](removeAt.apply(tensor.shape), indexOf.toInt, tensor)
  }

  def apply[S <: Shape, D](
      shape: S,
      values: Array[Float]
  )(implicit
      dataOps: DataOps[D]
  ): Tensor[S, D] = TConst(shape, dataOps, values)

  def param[S <: Shape, D](
      shape: S,
      values: () => D,
      update: D => Unit
  )(implicit
      dataOps: DataOps[D]
  ): Tensor[S, D] = TParam(shape, dataOps, values, update)

  // FUNCTIONS

  implicit def logTensor[S <: Shape, D]: log.Apply[Tensor[S, D], Tensor[S, D]] = new Functions.log.Apply[Tensor[S, D], Tensor[S, D]] {

    def apply(in: Tensor[S, D]): Tensor[S, D] = TLog(in)
  }

  implicit def expTensor[S <: Shape, D]: exp.Apply[Tensor[S, D], Tensor[S, D]] = new Functions.exp.Apply[Tensor[S, D], Tensor[S, D]] {

    def apply(in: Tensor[S, D]): Tensor[S, D] = TExp(in)
  }
}
