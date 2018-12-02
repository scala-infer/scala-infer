package scappla.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object Nd4jTensor {

  implicit val ops : DataOps[INDArray] = new DataOps[INDArray] {

    override def zeros(dims: Int*): INDArray =
      Nd4j.create(dims: _*)

    override def set(a: Array[Float], dims: Int*): INDArray =
      Nd4j.create(a, dims.toArray)

    override def get(a: INDArray): Array[Float] =
      a.data().asFloat()

    // elementwise operations

    override def plus(a: INDArray, b: INDArray): INDArray =
      a.add(b)

    override def minus(a: INDArray, b: INDArray): INDArray =
      a.sub(b)

    override def times(a: INDArray, b: INDArray): INDArray =
      a.mul(b)

    override def div(a: INDArray, b: INDArray): INDArray =
      a.div(b)

    override def negate(a: INDArray): INDArray =
      a.neg()

    override def log(a: INDArray): INDArray =
      Transforms.log(a)

    override def exp(a: INDArray): INDArray =
      Transforms.exp(a)

    // reshaping operations

    override def sum(a: INDArray, dim: Int): INDArray = {
      val oldShape = a.shape().toSeq
      val result = a.sum(dim)
      val newShape = oldShape.take(dim) ++ oldShape.drop(dim + 1)
      val finalResult = result.reshape(newShape: _*)
      finalResult
    }

    override def broadcast(a: INDArray, dimIndex: Int, dimSize: Int): INDArray = {
      val oldShape = a.shape().toSeq
      val newTmp: Seq[Long] = (oldShape.take(dimIndex) :+ 1L) ++ oldShape.drop(dimIndex)
      val reshaped = a.reshape(newTmp.toArray: _*)
      val newShape: Seq[Long] = (oldShape.take(dimIndex) :+ dimSize.toLong) ++ oldShape.drop(dimIndex)
      reshaped.broadcast(newShape.toArray: _*)
    }
  }

}

