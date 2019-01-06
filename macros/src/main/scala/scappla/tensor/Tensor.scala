package scappla.tensor

import scappla.Functions.{exp, log, pow, sum}
import scappla._
import scappla.distributions.RandomGaussian
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
}

case class TBuffer[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]])
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

case class TConst[S <: Shape, D: DataOps](override val v: Tensor[S, D])
    extends Constant[Tensor[S, D]](v)

case class TNeg[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = upstream.v.copy(
    data = upstream.v.dataOps.negate(upstream.v.data)
  )

  override def dv(dv: Tensor[S, D]): Unit = {
    upstream.dv(dv.copy(
      data = upstream.v.dataOps.negate(dv.data)
    ))
  }
}

case class TPlus[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
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

case class TMinus[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
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

case class TTimes[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
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

case class TDiv[S <: Shape, D: DataOps](numer: Expr[Tensor[S, D]], denom: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
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

case class TPow[S <: Shape, D: DataOps](base: Expr[Tensor[S, D]], expo: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
    val nt = base.v
    val dt = expo.v
    Tensor(nt.shape, nt.dataOps.pow(nt.data, dt.data))
  }

  override def dv(dx: Tensor[S, D]): Unit = {
    val shape = v.shape
    val ops = implicitly[DataOps[D]]
    base.dv(Tensor(
      shape,
      ops.times(
        ops.times(dx.data, expo.v.data),
        ops.pow(
          base.v.data,
          ops.minus(
            expo.v.data, ops.fill(1f, v.shape.sizes:_*)
          )
        )
      )
    ))
    expo.dv(Tensor(
      shape,
      ops.times(
        ops.times(dx.data, v.data),
        ops.log(base.v.data)
      )
    ))
  }
}

case class TLog[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]]) extends TensorExpr[S, D] {

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

case class TExp[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]]) extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    Tensor(ut.shape, ut.dataOps.exp(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val ut = upstream.v
    upstream.dv(Tensor(ut.shape, ut.dataOps.times(dv.data, ut.data)))
  }
}

case class TSum[R <: Shape, S <: Shape, D: DataOps](
    shape: R, index: Int, upstream: Expr[Tensor[S, D]]
) extends TensorExpr[R, D] {

  override val v: Tensor[R, D] = {
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

case class TSumAll[S <: Shape, D: DataOps](
    upstream: Expr[Tensor[S, D]]
) extends Real {

  override val v: Double = {
    upstream.v.dataOps.sumAll(upstream.v.data)
  }

  override def dv(v: Double): Unit = {
    val shape = upstream.v.shape
    upstream.dv(Tensor(shape, upstream.v.dataOps.fill(v.toFloat, shape.sizes: _*)))
  }
}

case class TBroadcast[S <: Shape, D: DataOps](
    upstream: Real, shape: S
) extends TensorExpr[S, D] {

  override val v: Tensor[S, D] = {
    val ops = implicitly[DataOps[D]]
    val data = ops.fill(upstream.v.toFloat, shape.sizes: _*)
    Tensor(shape, data)
  }

  override def dv(v: Tensor[S, D]): Unit = {
    val ops = implicitly[DataOps[D]]
    upstream.dv(ops.sumAll(v.data))
  }
}

object TensorExpr {

  implicit def toConst[S <: Shape, D: DataOps](
      tensor: Tensor[S, D]
  ): Expr[Tensor[S, D]] = TConst(tensor)

  def apply[S <: Shape, D: DataOps](
      shape: S, data: D
  ): Expr[Tensor[S, D]] = TConst(Tensor(shape, data))

  def sumAlong[S <: Shape, D <: Dim[_], I <: Nat, R <: Shape, X: DataOps](
      tensor: Expr[Tensor[S, X]]
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): TensorExpr[R, X] = {
    TSum[R, S, X](removeAt.apply(tensor.v.shape), indexOf.toInt, tensor)
  }

  def broadcast[S <: Shape, D: DataOps](
      real: Real, shape: S
  ): Expr[Tensor[S, D]] = TBroadcast(real, shape)

  def param[S <: Shape, D: DataOps](
      values: Tensor[S, D],
      update: Tensor[S, D] => Tensor[S, D]
  ): Expr[Tensor[S, D]] = TParam(values, update)

  // FUNCTIONS

  implicit def logTensor[S <: Shape, D: DataOps]: log.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]]] =
    new Functions.log.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]]] {
      def apply(in: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] = TLog(in)
    }

  implicit def expTensor[S <: Shape, D: DataOps]: exp.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]]] =
    new Functions.exp.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]]] {
      def apply(in: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] = TExp(in)
    }

  implicit def sumTensor[S <: Shape, D: DataOps]: sum.Apply[Expr[Tensor[S, D]]] =
    new sum.Apply[Expr[Tensor[S, D]]] {
      override def apply(in: Expr[Tensor[S, D]]): Real = TSumAll(in)
    }

  implicit def powTensor[S <: Shape, D: DataOps]: pow.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]], Expr[Tensor[S, D]]] =
    new pow.Apply[Expr[Tensor[S, D]], Expr[Tensor[S, D]], Expr[Tensor[S, D]]] {
      override def apply(base: Expr[Tensor[S, D]], exp: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] = TPow(base, exp)
    }

  implicit def numTensor[S <: Shape, D: DataOps] = new LiftedFractional[Tensor[S, D], S] {

    override def const(x: Tensor[S, D]): Expr[Tensor[S, D]] =
      TConst(x)

    override def div(x: Expr[Tensor[S, D]], y: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] =
      TDiv(x, y)

    override def plus(x: Expr[Tensor[S, D]], y: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] =
      TPlus(x, y)

    override def minus(x: Expr[Tensor[S, D]], y: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] =
      TMinus(x, y)

    override def times(x: Expr[Tensor[S, D]], y: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] =
      TTimes(x, y)

    override def negate(x: Expr[Tensor[S, D]]): Expr[Tensor[S, D]] =
      TNeg(x)

    override def fromInt(x: Int): Expr[Tensor[S, D]] = ???

    override def toInt(x: Expr[Tensor[S, D]]): Int = ???

    override def toLong(x: Expr[Tensor[S, D]]): Long = ???

    override def toFloat(x: Expr[Tensor[S, D]]): Float = ???

    override def toDouble(x: Expr[Tensor[S, D]]): Double = ???

    override def compare(x: Expr[Tensor[S, D]], y: Expr[Tensor[S, D]]): Int = ???

    override def fromInt(x: Int, shape: S): Expr[Tensor[S, D]] =
      broadcast(Real(x), shape)
  }

  implicit def shapeOf[S <: Shape, D: DataOps]: ShapeOf[Tensor[S, D], S] =
    new ShapeOf[Tensor[S, D], S] {
      override def apply(tensor: Tensor[S, D]): S = tensor.shape
    }

  implicit def gaussian[S <: Shape, D: DataOps](
      implicit so: ShapeOf[Tensor[S, D], S]
  ): RandomGaussian[Tensor[S, D], S] =
    new RandomGaussian[Tensor[S, D], S] {
      override def gaussian(shape: S): Tensor[S, D] = {
        val data = implicitly[DataOps[D]].gaussian(shape.sizes: _*)
        Tensor(shape, data)
      }
    }

}
