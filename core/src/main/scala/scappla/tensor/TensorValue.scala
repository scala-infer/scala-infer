package scappla.tensor

import scappla.Functions._
import scappla._
import shapeless.Nat
import Tensor._

trait TensorValue[S <: Shape, D] extends Value[Tensor[S, D]] {

  protected val ops: DataOps[D]

  override def buffer: TBuffer[S, D] = {
    TBuffer(this)(ops)
  }

  override def const: Constant[Tensor[S, D]] = {
    TConst(v)(ops)
  }
}

object TensorValue {

  implicit def toConst[S <: Shape, D: DataOps](
      tensor: Tensor[S, D]
  ): Value[Tensor[S, D]] = TConst(tensor)

  def apply[S <: Shape, D: DataOps](
      shape: S, data: D
  ): Value[Tensor[S, D]] = TConst(Tensor(shape, data))

  def count[S <: Shape, X : DataOps](
      tensor: Value[Tensor[S, X]], cond: Condition
  ): Int = {
    implicitly[DataOps[X]].count(tensor.v.data, cond)
  }

  def cumsum[S <: Shape, D <: Dim[_], I <: Nat, X: DataOps](
      tensor: Value[Tensor[S, X]], dim: D
  )(implicit
      indexOf: IndexOf.Aux[S, D, I]
  ): TensorValue[S, X] = {
    TCumSum(indexOf.toInt, tensor)
  }

  def sumAlong[S <: Shape, D <: Dim[_], I <: Nat, R <: Shape, X: DataOps](
      tensor: Value[Tensor[S, X]], dim: D
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): TensorValue[R, X] = {
    TSum[R, S, X](removeAt.apply(tensor.v.shape), indexOf.toInt, tensor)
  }

  def at[D: DataOps, S <: Shape](
      tensor: Value[Tensor[S, D]],
      index: Index[S]
  ): Value[Double] = {
    TAt[S, D](tensor, index)
  }

  def maxIndex[S <: Shape, D: DataOps](
      tensor: Value[Tensor[S, D]]
  ): Index[S] = {
    val indices = implicitly[DataOps[D]].imax(tensor.v.data)
    Index[S](indices.toList)
  }

  def broadcast[S <: Shape, D: DataOps](
      real: Real, shape: S
  ): Value[Tensor[S, D]] = TBroadcast(real, shape)

  def param[S <: Shape, D: DataOps](
      values: Tensor[S, D],
      update: Tensor[S, D] => Tensor[S, D]
  ): Value[Tensor[S, D]] = TParam(values, update)

  // FUNCTIONS

  implicit def logTensor[S <: Shape, D: DataOps]: log.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] =
    new Functions.log.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] {
      def apply(in: Value[Tensor[S, D]]): Value[Tensor[S, D]] = TLog(in)
    }

  implicit def expTensor[S <: Shape, D: DataOps]: exp.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] =
    new Functions.exp.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] {
      def apply(in: Value[Tensor[S, D]]): Value[Tensor[S, D]] = TExp(in)
    }

  implicit def tanhTensor[S <: Shape, D: DataOps](
    implicit field: TensorExprField[S, D]
  ): tanh.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] =
    new tanh.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] {
      override def apply(x: Value[Tensor[S, D]]) = {
        val shape = x.v.shape
        val one = field.fromInt(1, shape)
        val two = field.fromInt(2, shape)
        TPlus(field.negate(one), TDiv(two, TPlus(TExp(TNeg(TTimes(two, x))), one)))
      }
    }

  implicit def sigmoidTensor[S <: Shape, D: DataOps](
    implicit field: TensorExprField[S, D]
  ): sigmoid.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] =
    new Functions.sigmoid.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]]] {
      override def apply(x: Value[Tensor[S, D]]): Value[Tensor[S, D]] =  {
        val shape = x.v.shape
        val one = field.fromInt(1, shape)
        TDiv(one, TPlus(TExp(field.negate(x)), one))
      }
    }

  implicit def sumTensor[S <: Shape, D: DataOps]: sum.Apply[Value[Tensor[S, D]], Value[Double]] =
    new sum.Apply[Value[Tensor[S, D]], Value[Double]] {
      override def apply(in: Value[Tensor[S, D]]): Real = TSumAll(in)
    }

  implicit def powTensor[S <: Shape, D: DataOps]: pow.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]], Value[Tensor[S, D]]] =
    new pow.Apply[Value[Tensor[S, D]], Value[Tensor[S, D]], Value[Tensor[S, D]]] {
      override def apply(base: Value[Tensor[S, D]], exp: Value[Tensor[S, D]]): Value[Tensor[S, D]] = TPow(base, exp)
    }

  implicit def numTensorExpr[S <: Shape, D: DataOps] = new TensorExprField[S, D]

  class TensorExprField[S <: Shape, D: DataOps] extends ValueField[Tensor[S, D], S] {

    private implicit val baseField = implicitly[TensorField[S, D]]

    override def div(x: Value[Tensor[S, D]], y: Value[Tensor[S, D]]): Value[Tensor[S, D]] =
      TDiv(x, y)

    override def plus(x: Value[Tensor[S, D]], y: Value[Tensor[S, D]]): Value[Tensor[S, D]] =
      TPlus(x, y)

    override def minus(x: Value[Tensor[S, D]], y: Value[Tensor[S, D]]): Value[Tensor[S, D]] =
      TMinus(x, y)

    override def times(x: Value[Tensor[S, D]], y: Value[Tensor[S, D]]): Value[Tensor[S, D]] =
      TTimes(x, y)

    override def negate(x: Value[Tensor[S, D]]): Value[Tensor[S, D]] =
      TNeg(x)

    override def fromInt(x: Int): Value[Tensor[S, D]] = ???

    override def toInt(x: Value[Tensor[S, D]]): Int = ???

    override def toLong(x: Value[Tensor[S, D]]): Long = ???

    override def toFloat(x: Value[Tensor[S, D]]): Float = ???

    override def toDouble(x: Value[Tensor[S, D]]): Double = ???

    override def compare(x: Value[Tensor[S, D]], y: Value[Tensor[S, D]]): Int = ???

    override def const(x: Tensor[S, D]): Value[Tensor[S, D]] =
      TConst(x)

    override def fromInt(x: Int, shape: S): Value[Tensor[S, D]] =
      broadcast(Real(x), shape)

    override def buffer(ex: Value[Tensor[S, D]]) =
      TBuffer(ex)

    override implicit def mkNumericOps(expr: Value[Tensor[S, D]]) = new TensorExprOps(expr)

    class TensorExprOps(lhs: Value[Tensor[S, D]]) extends FractionalOps(lhs) {

      def :*:[T <: Shape, R <: Shape](
          rhs: Value[Tensor[T, D]]
      )(implicit
          sd: SymDiff.Aux[S, T, R]
      ): Value[Tensor[R, D]] = new Value[Tensor[R, D]] {

        // S :*: T => R
        override def v: Tensor[R, D] = {
          rhs.v :*: lhs.v
        }

        override def dv(v: Tensor[R, D]): Unit = {
          implicit val leftSd = sd.recoverLeft
          lhs.dv(v :*: rhs.v)

          implicit val rightSd = sd.recoverRight
          rhs.dv(lhs.v :*: v)
        }
      }
    }
  }

  implicit def infixTensorExprOps[S <: Shape, D: DataOps](expr: Value[Tensor[S, D]])(implicit num: TensorExprField[S, D]) = new num.TensorExprOps(expr)

}

case class TBuffer[S <: Shape, D: DataOps](upstream: Value[Tensor[S, D]])
    extends TensorValue[S, D] with Buffered[Tensor[S, D]] {

  private var refCount: Int = 1

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  private var grad: Option[D] = None

  override def v: Tensor[S, D] = upstream.v

  override def dv(gradient: Tensor[S, D]): Unit = {
    grad = grad.map {
      ops.plus(_, gradient.data)
    }.orElse(Some(gradient.data))
  }

  /**
    * basic refcounting; when a function needs a buffer instead of a plain Real
    * (i.e. when it has potentially more than one use of the value), it further defers
    * the backpropagation of the gradient.  When all references have completed, can the
    * gradient be propagated further backwards.
    */
  override def buffer: TBuffer[S, D] = {
    refCount += 1
    this
  }

  override def complete(): Unit = {
    refCount -= 1
    if (refCount == 0) {
      grad.foreach { g =>
        upstream.dv(Tensor(v.shape, g))
      }
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
) extends TensorValue[S, D] {

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

case class TNeg[S <: Shape, D: DataOps](upstream: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TPlus[S <: Shape, D: DataOps](left: Value[Tensor[S, D]], right: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TMinus[S <: Shape, D: DataOps](left: Value[Tensor[S, D]], right: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TTimes[S <: Shape, D: DataOps](left: Value[Tensor[S, D]], right: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TDiv[S <: Shape, D: DataOps](numer: Value[Tensor[S, D]], denom: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TPow[S <: Shape, D: DataOps](base: Value[Tensor[S, D]], expo: Value[Tensor[S, D]])
    extends TensorValue[S, D] {

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

case class TLog[S <: Shape, D: DataOps](upstream: Value[Tensor[S, D]]) extends TensorValue[S, D] {

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

case class TExp[S <: Shape, D: DataOps](upstream: Value[Tensor[S, D]]) extends TensorValue[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    Tensor(ut.shape, ops.exp(ut.data))
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val tv = this.v
    upstream.dv(Tensor(tv.shape, ops.times(dv.data, tv.data)))
  }

  override def toString: String = {
    s"exp($upstream)"
  }
}

case class TSum[R <: Shape, S <: Shape, D: DataOps](
    shape: R, index: Int, upstream: Value[Tensor[S, D]]
) extends TensorValue[R, D] {

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

case class TAt[S <: Shape, D: DataOps](
    upstream: Value[Tensor[S, D]],
    index: Index[S]
) extends Value[Double] {

  override val v: Double = {
    implicitly[DataOps[D]].get(upstream.v.data, index.indices: _*)
  }

  override def dv(dv: Double): Unit = {
    val ops = implicitly[DataOps[D]]
    val shape = upstream.v.shape
    val tensorData = ops.fill(0f, shape.sizes: _*)
    ops.put(tensorData, dv.toFloat, index.indices: _*)
    upstream.dv(
      Tensor(
        shape,
        tensorData
      )
    )
  }
}

case class TCumSum[S <: Shape, D: DataOps](
    index: Int, upstream: Value[Tensor[S, D]]
) extends TensorValue[S, D] {

  override val ops: DataOps[D] = implicitly[DataOps[D]]

  override val v: Tensor[S, D] = {
    val ut = upstream.v
    Tensor(
      upstream.v.shape,
      ops.cumsum(ut.data, index)
    )
  }

  override def dv(dv: Tensor[S, D]): Unit = {
    val summed_dv = ops.sum(dv.data, index)
    val sum_s = ops.broadcast(summed_dv, index, dv.shape.sizes(index))

    val ut = upstream.v
    upstream.dv(Tensor(
      ut.shape,
      ops.minus(sum_s, ops.minus(ops.cumsum(dv.data, index), dv.data))
    ))
  }

  override def toString: String = {
    s"cumsum($upstream, $index)"
  }
}

case class TSumAll[S <: Shape, D: DataOps](
    upstream: Value[Tensor[S, D]]
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
) extends TensorValue[S, D] {

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