package scappla.tensor

import scappla.Functions._
import scappla._

import shapeless.Nat

object Tensor {

  implicit def numTensor[D: TensorData, S <: Shape] = new TensorField[D, S]

  implicit def sumTensor[D: TensorData, S <: Shape]: sum.Apply[Value[D, S], Value[Double, Unit]] =
    new sum.Apply[Value[D, S], Value[Double, Unit]] {
      override def apply(in: Value[D, S]): Real = TSumAll(in)
    }

  case class TSumAll[D: TensorData, S <: Shape](
      upstream: Value[D, S]
  ) extends AbstractReal {

    private val ops = implicitly[TensorData[D]]

    override val v: Double = {
      ops.sumAll(upstream.v)
    }

    override def dv(v: Double): Unit = {
      val shape = upstream.shape
      val data = ops.fill(v.toFloat, shape.sizes: _*)
      upstream.dv(data)
    }

    override def toString: String = {
      s"sumAll($upstream)"
    }
  }

  def sumAlong[S <: Shape, D <: Dim[_], I <: Nat, R <: Shape, X: TensorData](
      tensor: Value[X, S], dim: D
  )(implicit
      indexOf: IndexOf.Aux[S, D, I],
      removeAt: RemoveAt.Aux[S, I, R]
  ): Value[X, R] = {
    TSum[R, S, X](removeAt.apply(tensor.shape), indexOf.toInt, tensor)
  }

  case class TSum[R <: Shape, S <: Shape, D: TensorData](
      shape: R, index: Int, upstream: Value[D, S]
  ) extends Value[D, R] {

    private val ops: TensorData[D] = implicitly[TensorData[D]]

    override def field = implicitly[TensorField[D, R]]

    override val v: D = {
      ops.sum(upstream.v, index)
    }

    override def dv(dv: D): Unit = {
      val us = upstream.shape
      upstream.dv(
        ops.broadcast(dv, index, us.sizes(index))
      )
    }

    override def toString: String = {
      s"sum($upstream, $index)"
    }
  }

  def cumsum[S <: Shape, D <: Dim[_], I <: Nat, X: TensorData](
      tensor: Value[X, S], dim: D
  )(implicit
      indexOf: IndexOf.Aux[S, D, I]
  ): Value[X, S] = {
    TCumSum(indexOf.toInt, tensor)
  }

  case class TCumSum[S <: Shape, D <: Dim[_], I <: Nat, X: TensorData](
      index: Int, upstream: Value[X, S]
  ) extends Value[X, S] {

    private val ops: TensorData[X] = implicitly[TensorData[X]]

    override def field = upstream.field

    override def shape = upstream.shape

    override val v: X = {
      ops.cumsum(upstream.v, index)
    }

    override def dv(dv: X): Unit = {
      val summed_dv = ops.sum(dv, index)
      val sum_s = ops.broadcast(summed_dv, index, shape.sizes(index))
      val ut = upstream.v
      upstream.dv(
        ops.minus(sum_s, ops.minus(ops.cumsum(dv, index), dv))
      )
    }

    override def toString: String = {
      s"cumsum($upstream, $index)"
    }
  }

  def at[D: TensorData, S <: Shape](
      tensor: Value[D, S],
      index: Index[S]
  ): Value[Double, Unit] = {
    TAt[S, D](tensor, index)
  }

  case class TAt[S <: Shape, D: TensorData](
      upstream: Value[D, S],
      index: Index[S]
  ) extends AbstractReal {

    private val ops: TensorData[D] = implicitly[TensorData[D]]

    override val v: Double = {
      ops.get(upstream.v, index.indices: _*)
    }

    override def dv(dv: Double): Unit = {
      val shape = upstream.shape
      val tensorData = ops.fill(0f, shape.sizes: _*)
      ops.put(tensorData, dv.toFloat, index.indices: _*)
      upstream.dv(tensorData)
    }
  }

  def broadcast[S <: Shape, D: TensorData](
      real: Real, shape: S
  ): Value[D, S] = TBroadcast(real, shape)

  case class TBroadcast[S <: Shape, D: TensorData](
      upstream: Real, shape: S
  ) extends Value[D, S] {

    private val ops: TensorData[D] = implicitly[TensorData[D]]

    override def field = implicitly[TensorField[D, S]]

    override val v: D = {
      ops.fill(upstream.v.toFloat, shape.sizes: _*)
    }

    override def dv(v: D): Unit = {
      upstream.dv(ops.sumAll(v))
    }

    override def toString: String = {
      s"broadcast($upstream, $shape)"
    }
  }

  def maxIndex[S <: Shape, D: TensorData](
      tensor: Value[D, S]
  ): Index[S] = {
    val indices = implicitly[TensorData[D]].imax(tensor.v)
    Index[S](indices.toList)
  }

  def count[S <: Shape, X : TensorData](
      tensor: Value[X, S], cond: Condition
  ): Int = {
    implicitly[TensorData[X]].count(tensor.v, cond)
  }

  // implicit def numTensorExpr[S <: Shape, D: DataOps] = TensorExpr.numTensorExpr[S, D]

  class TensorField[D: TensorData, S <: Shape] extends BaseField[D, S] {

    private val ops = implicitly[TensorData[D]]

    override def fromInt(x: Int, shape: S): D = {
      ops.fill(x.toFloat, shape.sizes: _*)
    }

    override def fromDouble(x: Double, shape: S): D = {
      ops.fill(x.toFloat, shape.sizes: _*)
    }

    override def sqrt(tensor: D): D = {
      ops.sqrt(tensor)
    }

    override def gaussian(shape: S): D = {
      ops.gaussian(shape.sizes: _*)
    }

    override def div(x: D, y: D): D = {
      ops.div(x, y)
    }

    override def plus(x: D, y: D): D = {
      ops.plus(x, y)
    }

    override def minus(x: D, y: D): D = {
      ops.minus(x, y)
    }

    override def times(x: D, y: D): D = {
      ops.times(x, y)
    }

    override def negate(x: D): D = {
      ops.negate(x)
    }

    override def exp(x: D):D = {
      ops.exp(x)
    }

    override def log(x: D):D = {
      ops.log(x)
    }

    override def sigmoid(x: D):D = {
      ops.sigmoid(x)
    }

    override def pow(x: D, y: D):D = {
      ops.pow(x, y)
    }

    def sumAll(x: D): Float = {
      ops.sumAll(x)
    }
  }

  def tensordot[D: TensorData, S <: Shape, T <: Shape, R <: Shape](lhs: D, rhs: D)(implicit
      sd: SymDiff.Aux[S, T, R]
    ): D = {
      val ops = implicitly[TensorData[D]]
      ops.tensordot(
        lhs,
        rhs,
        sd.matchedIndices,              // (S T) R
        sd.recoverLeft.matchedIndices,  // (T R) S
        sd.recoverRight.matchedIndices  // (R S) T
      )
    }

  implicit def infixTensorOps[S <: Shape, D: TensorData](lhs: Value[D, S]) =
    new TensorOps(lhs)

  class TensorOps[D: TensorData, S <: Shape](lhs: Value[D, S]) {

    def :*:[T <: Shape, R <: Shape](rhs: Value[D, T])(implicit sd: SymDiff.Aux[S, T, R]): Value[D, R] =
      new Value[D, R] {

        override val field = implicitly[TensorField[D, R]]

        override val shape: R = sd.mapper.ab(lhs.shape, rhs.shape)

        // S :*: T => R
        override def v: D = {
//          println(s"LHS: ${lhs.shape}, RHS: ${rhs.shape}, OUT: ${shape}")
          tensordot(lhs.v, rhs.v)
        }

        override def dv(dv: D): Unit = {
          implicit val leftSd = sd.recoverLeft
//          println(s"LHS: ${rhs.shape}, RHS: ${shape}, OUT: ${lhs.shape}")
          val ldv = tensordot(rhs.v, dv)(implicitly[TensorData[D]], leftSd)
          lhs.dv(ldv)

          implicit val rightSd = sd.recoverRight
//          println(s"LHS: ${shape}, RHS: ${lhs.shape}, OUT: ${rhs.shape}")
          val rdv = tensordot(dv, lhs.v)(implicitly[TensorData[D]], rightSd)
          rhs.dv(rdv)
        }
      }
  }

  implicit def infixTensorExprOps[S <: Shape, D: TensorData](lhs: Expr[D, S]) =
    new TensorExprOps(lhs)

  class TensorExprOps[S <: Shape, D: TensorData](lhs: Expr[D, S]) {
      def :*:[T <: Shape, R <: Shape](rhs: Expr[D, T])
          (implicit sd: SymDiff.Aux[S, T, R]): Expr[D, R] =
        Apply2(lhs, rhs, { (lv: Value[D, S], rv: Value[D, T]) =>
          lv.:*:(rv)
        })
  }
}