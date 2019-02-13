package scappla.tensor

import scappla.Functions.{exp, log, pow, sum}
import scappla._
import shapeless.Nat


case class Tensor[S <: Shape, D](shape: S, data: D)

trait TensorExpr[S <: Shape, D] extends Expr[Tensor[S, D]] {

  protected val ops: DataOps[D]

  override def buffer: TBuffer[S, D] = {
    TBuffer(this)(ops)
  }

  override def const: Constant[Tensor[S, D]] = {
    TConst(v)(ops)
  }
}

case class TBuffer[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] with Buffered[Tensor[S, D]] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  private var grad: Option[D] = None

  override val v: Tensor[S, D] = upstream.v

  override def dv(gradient: Tensor[S, D]): Unit = {
    grad = grad.map {
      ops.plus(_, gradient.data)
    }.orElse(Some(gradient.data))
  }

  override def complete(): Unit = {
    grad.foreach { g =>
      upstream.dv(Tensor(v.shape, g))
    }
    grad = None
  }

  override def toString: String = {
    s"Buffer($upstream)"
  }
}

case class TParam[S <: Shape, D: DataOps](
    var v: Tensor[S, D],
    update: Tensor[S, D] => Tensor[S, D]
) extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override def dv(gradient: Tensor[S, D]): Unit =
    v = update(gradient)

  override def toString: String = {
    "Param"
  }
}

case class TConst[S <: Shape, D: DataOps](override val v: Tensor[S, D])
    extends Constant[Tensor[S, D]](v) {

  override def toString: String = {
    s"const(${v.hashCode()})"
  }
}

case class TNeg[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = upstream.v.copy(
    data = ops.negate(upstream.v.data)
  )

  override def dv(dv: Tensor[S, D]): Unit = {
    upstream.dv(dv.copy(
      data = ops.negate(dv.data)
    ))
  }

  override def toString: String = {
    s"- $upstream"
  }
}

case class TPlus[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)

    Tensor(lt.shape, ops.plus(lt.data, rt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    left.dv(dv)
    right.dv(dv)
  }

  override def toString: String = {
    s"($left + $right)"
  }
}

case class TMinus[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)

    Tensor(lt.shape, ops.minus(lt.data, rt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    left.dv(dv)
    right.dv(dv.copy(data = ops.negate(dv.data)))
  }

  override def toString: String = {
    s"($left - $right)"
  }
}

case class TTimes[S <: Shape, D: DataOps](left: Expr[Tensor[S, D]], right: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val lt = left.v
    val rt = right.v
    assert(lt.shape == rt.shape)

    Tensor(lt.shape, ops.times(lt.data, rt.data))
  }

  override def dv(gradient: Tensor[S, D]): Unit = {
    left.dv(gradient.copy(
      data = ops.times(gradient.data, right.v.data)
    ))
    right.dv(gradient.copy(
      data = ops.times(gradient.data, left.v.data)
    ))
  }

  override def toString: String = {
    s"($left * $right)"
  }
}

case class TDiv[S <: Shape, D: DataOps](numer: Expr[Tensor[S, D]], denom: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val nt = numer.v
    val dt = denom.v
    Tensor(nt.shape, ops.div(nt.data, dt.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val shape = v.shape
    val dt = dv.data
    val dent = denom.v.data

    numer.dv(Tensor(shape, ops.div(dt, dent)))
    denom.dv(Tensor(shape, ops.div(
      ops.times(dt, v.data),
      ops.negate(dent)
    )))
  }

  override def toString: String = {
    s"($numer / $denom)"
  }
}

case class TPow[S <: Shape, D: DataOps](base: Expr[Tensor[S, D]], expo: Expr[Tensor[S, D]])
    extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val nt = base.v
    val dt = expo.v
    Tensor(nt.shape, ops.pow(nt.data, dt.data))
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
            expo.v.data, ops.fill(1f, v.shape.sizes: _*)
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

  override def toString: String = {
    s"($base ^ $expo)"
  }
}

case class TLog[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]]) extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    ut.copy(data = ops.log(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val ut = upstream.v
    upstream.dv(
      Tensor(ut.shape, ops.div(dv.data, ut.data))
    )
  }

  override def toString: String = {
    s"log($upstream)"
  }
}

case class TExp[S <: Shape, D: DataOps](upstream: Expr[Tensor[S, D]]) extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    Tensor(ut.shape, ops.exp(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val ut = upstream.v
    upstream.dv(Tensor(ut.shape, ops.times(dv.data, ut.data)))
  }

  override def toString: String = {
    s"exp($upstream)"
  }
}

case class TSum[R <: Shape, S <: Shape, D: DataOps](
    shape: R, index: Int, upstream: Expr[Tensor[S, D]]
) extends TensorExpr[R, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[R, D] = {
    val ut = upstream.v
    Tensor(shape,
      ops.sum(ut.data, index)
    )
  }

  override def dv(dv: Tensor[R, D]): Unit = {
    val ut = upstream.v
    upstream.dv(Tensor(ut.shape,
      ops.broadcast(dv.data, index, ut.shape.sizes(index))
    ))
  }

  override def toString: String = {
    s"sum($upstream, $index)"
  }
}

case class TSumAll[S <: Shape, D: DataOps](
    upstream: Expr[Tensor[S, D]]
) extends Real {

  private val ops = implicitly[DataOps[D]]

  override val v: Double = {
    ops.sumAll(upstream.v.data)
  }

  override def dv(v: Double): Unit = {
    val shape = upstream.v.shape
    val data = ops.fill(v.toFloat, shape.sizes: _*)
    upstream.dv(Tensor(shape, data))
  }

  override def toString: String = {
    s"sumAll($upstream)"
  }
}

case class TBroadcast[S <: Shape, D: DataOps](
    upstream: Real, shape: S
) extends TensorExpr[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val data = ops.fill(upstream.v.toFloat, shape.sizes: _*)
    Tensor(shape, data)
  }

  override def dv(v: Tensor[S, D]): Unit = {
    upstream.dv(ops.sumAll(v.data))
  }

  override def toString: String = {
    s"broadcast($upstream, $shape)"
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

  implicit def numTensorExpr[S <: Shape, D: DataOps] = new InferField[Tensor[S, D], S] {

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

    override def const(x: Tensor[S, D]): Expr[Tensor[S, D]] =
      TConst(x)

    override def fromInt(x: Int, shape: S): Expr[Tensor[S, D]] =
      broadcast(Real(x), shape)
  }

  implicit def numTensor[S <: Shape, D: DataOps] = new BaseField[Tensor[S, D], S] {

    private val ops = implicitly[DataOps[D]]

    override def shapeOf(tensor: Tensor[S, D]): S = tensor.shape

    override def fromInt(x: Int, shape: S): Tensor[S, D] = {
      val data = ops.fill(x.toFloat, shape.sizes: _*)
      Tensor(shape, data)
    }

    override def fromDouble(x: Double, shape: S): Tensor[S, D] = {
      val data = ops.fill(x.toFloat, shape.sizes: _*)
      Tensor(shape, data)
    }

    override def sqrt(tensor: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.sqrt(tensor.data)
      Tensor(tensor.shape, data)
    }

    override def gaussian(shape: S): Tensor[S, D] = {
      val data = ops.gaussian(shape.sizes: _*)
      Tensor(shape, data)
    }

    override def div(x: Tensor[S, D], y: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.div(x.data, y.data)
      Tensor(x.shape, data)
    }

    override def plus(x: Tensor[S, D], y: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.plus(x.data, y.data)
      Tensor(x.shape, data)
    }

    override def minus(x: Tensor[S, D], y: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.minus(x.data, y.data)
      Tensor(x.shape, data)
    }

    override def times(x: Tensor[S, D], y: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.times(x.data, y.data)
      Tensor(x.shape, data)
    }

    override def negate(x: Tensor[S, D]): Tensor[S, D] = {
      val data = ops.negate(x.data)
      Tensor(x.shape, data)
    }

    override def fromInt(x: Int): Tensor[S, D] = ???

    override def toInt(x: Tensor[S, D]): Int = ???

    override def toLong(x: Tensor[S, D]): Long = ???

    override def toFloat(x: Tensor[S, D]): Float = ???

    override def toDouble(x: Tensor[S, D]): Double = ???

    override def compare(x: Tensor[S, D], y: Tensor[S, D]): Int = ???
  }

  implicit def tensorOps[S <: Shape, D: DataOps](expr: Expr[Tensor[S, D]]) =
    new TensorOps(expr)

  class TensorOps[S <: Shape, D: DataOps](expr: Expr[Tensor[S, D]]) {

    def :*:[T <: Shape, R <: Shape](
        other: Expr[Tensor[T, D]]
    )(implicit
        sd: SymDiff.Aux[S, T, R]
    ): Expr[Tensor[R, D]] = new Expr[Tensor[R, D]] {

      override def v: Tensor[R, D] = {
        val tensor = expr.v
        val ops = implicitly[DataOps[D]]
        Tensor(
          sd.mapper.ab(expr.v.shape, other.v.shape),
          ops.einsum(tensor.data, other.v.data, sd.matchedIndices: _*)
        )
      }

      override def dv(v: Tensor[R, D]): Unit = {
        val ops = implicitly[DataOps[D]]
        expr.dv(
          Tensor(
            sd.mapper.bc(other.v.shape, v.shape),
            ops.einsum(other.v.data, v.data, sd.recoverLeft.matchedIndices: _*)
          )
        )
        other.dv(
          Tensor(
            sd.mapper.ca(v.shape, expr.v.shape),
            ops.einsum(v.data, expr.v.data, sd.recoverRight.matchedIndices: _*)
          )
        )
      }
    }
  }

}
