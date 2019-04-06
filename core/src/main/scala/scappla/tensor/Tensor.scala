package scappla.tensor

import scappla.Functions._
import scappla._

case class Tensor[S <: Shape, D](shape: S, data: D)

object Tensor {


  // implicit def numTensorExpr[S <: Shape, D: DataOps] = TensorExpr.numTensorExpr[S, D]

  implicit def numTensor[S <: Shape, D: DataOps] = new TensorField[S, D]

  class TensorField[S <: Shape, D: DataOps] extends BaseField[Tensor[S, D], S] {

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

    class TensorOps(lhs: Tensor[S, D]) extends FractionalOps(lhs) {

      def :*:[T <: Shape, R <: Shape](
        rhs: Tensor[T, D]
      )(implicit
        sd: SymDiff.Aux[S, T, R]
      ): Tensor[R, D] = {
        val ops = implicitly[DataOps[D]]
        Tensor(
          sd.mapper.ab(lhs.shape, rhs.shape),
          ops.einsum(
            lhs.data,
            rhs.data,
            sd.matchedIndices,              // (S T) R
            sd.recoverLeft.matchedIndices,  // (T R) S
            sd.recoverRight.matchedIndices  // (R S) T
          )
        )
      }
    }

    override implicit def mkNumericOps(lhs: Tensor[S, D]): TensorOps =
      new TensorOps(lhs)
  }

  implicit def infixTensorOps[S <: Shape, D: DataOps](lhs: Tensor[S, D])(implicit num: TensorField[S, D]) =
    new num.TensorOps(lhs)
}