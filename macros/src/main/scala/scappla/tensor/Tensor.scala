package scappla.tensor

import scappla.Functions.{exp, log}
import scappla.{Buffered, Expr, Functions}
import shapeless.Nat


case class Tensor[S <: Shape, D: DataOps](shape: S, data: D) {
  val dataOps: DataOps[D] = implicitly[DataOps[D]]
}

trait TensorExpr[S <: Shape, D] extends Expr[Tensor[S, D]] {

  @inline
  private implicit def ops = v.dataOps

  override def buffer: TBuffer[S, D] = {
    TBuffer(this)
  }

  def plus(other: TensorExpr[S, D]): TensorExpr[S, D] = {
    TPlus(this, other)
  }

  def minus(other: TensorExpr[S, D]): TensorExpr[S, D] = {
    TMinus(this, other)
  }

  def times(other: TensorExpr[S, D]): TensorExpr[S, D] = {
    TTimes(this, other)
  }

  def div(other: TensorExpr[S, D]): TensorExpr[S, D] = {
    TDiv(this, other)
  }

  def negate: TensorExpr[S, D] = {
    TNeg(this)
  }

}

case class TBuffer[S <: Shape, D: DataOps](upstream: TensorExpr[S, D])
    extends TensorExpr[S, D] with Buffered[Tensor[S, D]] {

  private var grad: Option[D] = None

  override val v: Tensor[S, D] = upstream.v

  override def dv(gradient: Tensor[S, D]): Unit = {
    grad = grad.map {
      gradient.dataOps.plus(_, gradient.data)
    }.orElse(Some(gradient.data))
  }

  override def complete(): Unit = {
    grad.foreach { g =>
      upstream.dv(Tensor(v.shape, g))
    }
    grad = None
  }
}

case class TParam[S <: Shape, D: DataOps](
    var v: Tensor[S, D],
    update: Tensor[S, D] => Tensor[S, D]
) extends TensorExpr[S, D] {
  override def dv(gradient: Tensor[S, D]): Unit =
    v = update(gradient)
}

case class TConst[S <: Shape, D: DataOps](v: Tensor[S, D]) extends TensorExpr[S, D] {
  override def dv(gradient: Tensor[S, D]): Unit = {}
}

case class TNeg[S <: Shape, D: DataOps](upstream: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = upstream.v.copy(
    data = upstream.v.dataOps.negate(upstream.v.data)
  )

  override def dv(dv: Tensor[S, D]): Unit = {
    upstream.dv(dv.copy(
      data = upstream.v.dataOps.negate(dv.data)
    ))
  }
}

case class TPlus[S <: Shape, D: DataOps](left: TensorExpr[S, D], right: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override def v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)
    assert(lt.dataOps == rt.dataOps)

    Tensor(lt.shape, lt.dataOps.plus(lt.data, rt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    left.dv(dv)
    right.dv(dv)
  }
}

case class TMinus[S <: Shape, D: DataOps](left: TensorExpr[S, D], right: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override def v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)
    assert(lt.dataOps == rt.dataOps)

    Tensor(lt.shape, lt.dataOps.minus(lt.data, rt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    left.dv(dv)
    right.dv(dv.copy(data = dv.dataOps.negate(dv.data)))
  }
}

case class TTimes[S <: Shape, D: DataOps](left: TensorExpr[S, D], right: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override def v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)

    Tensor(lt.shape, lt.dataOps.times(lt.data, rt.data))
  }

  override def dv(gradient: Tensor[S, D]): Unit = {
    left.dv(gradient.copy(
      data = gradient.dataOps.times(gradient.data, right.v.data)
    ))
    right.dv(gradient.copy(
      data = gradient.dataOps.times(gradient.data, left.v.data)
    ))
  }
}

case class TDiv[S <: Shape, D: DataOps](numer: TensorExpr[S, D], denom: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override def v: Tensor[S, D] = {
    val nt = numer.v
    val dt = denom.v
    Tensor(nt.shape, nt.dataOps.div(nt.data, dt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val shape = v.shape
    val ops = v.dataOps
    val nt = numer.v
    val dt = denom.v

    numer.dv(Tensor(shape, ops.div(dv.data, dt.data)))
    denom.dv(Tensor(shape, ops.div(
      ops.times(dv.data, nt.data),
      ops.times(dt.data, dt.data)
    )))
  }
}

case class TLog[S <: Shape, D: DataOps](upstream: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    ut.copy(data = ut.dataOps.log(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val ut = upstream.v
    upstream.dv(
      Tensor(ut.shape, ut.dataOps.div(dv.data, ut.data))
    )
  }
}

case class TExp[S <: Shape, D: DataOps](upstream: TensorExpr[S, D]) extends TensorExpr[S, D] {

  override def v: Tensor[S, D] = {
    val ut = upstream.v
    Tensor(ut.shape, ut.dataOps.exp(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val ut = upstream.v
    upstream.dv(Tensor(ut.shape, ut.dataOps.times(dv.data, ut.data)))
  }
}

case class TSum[R <: Shape, S <: Shape, D: DataOps](
    shape: R, index: Int, upstream: TensorExpr[S, D]
) extends TensorExpr[R, D] {

  override def v: Tensor[R, D] = {
    val ut = upstream.v
    Tensor(shape,
      ut.dataOps.sum(ut.data, index, ut.shape.sizes: _*)
    )
  }

  override def dv(dv: Tensor[R, D]): Unit = {
    val ut = upstream.v
    upstream.dv(Tensor(ut.shape,
      ut.dataOps.broadcast(dv.data, index, ut.shape.sizes(index), dv.shape.sizes: _*)
    ))
  }
}

object TensorExpr {

  def sum[S <: Shape, D <: Dim[_], I <: Nat, R <: Shape, X: DataOps](
      tensor: TensorExpr[S, X]
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): TensorExpr[R, X] = {
    TSum[R, S, X](removeAt.apply(tensor.v.shape), indexOf.toInt, tensor)
  }

  def apply[S <: Shape, D: DataOps](shape: S, data: D): TensorExpr[S, D] =
    TConst(Tensor(shape, data))

  def param[S <: Shape, D: DataOps](
      values: Tensor[S, D],
      update: Tensor[S, D] => Tensor[S, D]
  ): TensorExpr[S, D] = TParam(values, update)

  // FUNCTIONS

  implicit def logTensor[S <: Shape, D: DataOps]: log.Apply[TensorExpr[S, D], TensorExpr[S, D]] =
    new Functions.log.Apply[TensorExpr[S, D], TensorExpr[S, D]] {
      def apply(in: TensorExpr[S, D]): TensorExpr[S, D] = TLog(in)
    }

  implicit def expTensor[S <: Shape, D: DataOps]: exp.Apply[TensorExpr[S, D], TensorExpr[S, D]] =
    new Functions.exp.Apply[TensorExpr[S, D], TensorExpr[S, D]] {
      def apply(in: TensorExpr[S, D]): TensorExpr[S, D] = TExp(in)
    }

}
