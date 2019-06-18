package scappla.tensor.nd4j

import scappla._
import tensor._
import Tensor._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec

class Nd4jTensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "work with nd4j as well" in {
    implicit val nd4jTensor = Nd4jTensor.ops

    val shape = Batch(2)

    val data = Nd4j.create(
      Array(0.0f, 1.0f),
      shape.sizes.toArray
    )

    val tensor: Value[INDArray, Batch] = Value(data, shape)
    val sum = sumAlong(tensor, shape)
    val result = sum.v.data().asFloat()(0)

    assert(result == 1f)
  }

  it should "use tensordot for nd4j as well" in {
    implicit val nd4jTensor = Nd4jTensor.ops

    val inputDim = Input(1)
    val batchDim = Batch(2)
    val outDim = Other(3)
    val inputShape = inputDim :#: batchDim
    val outShape = batchDim :#: outDim

    val data: INDArray = Nd4j.create(
      Array(0f, 1f),
      inputShape.sizes.toArray
    )
    val input = Value(data, inputShape)

    val matrix = Nd4j.create(
      Array(
        0f, 2f, 4f,
        1f, 3f, 5f
      ),
      outShape.sizes.toArray
    )
    val param = TParam(matrix, outShape)

    val out: Value[INDArray, Other :#: Input] = input :*: param
    val outData = out.v
    val expected = ArrayTensor(Seq(3, 1), Array(1f, 3f, 5f))
    assert(outData.shape().map { _.toInt } sameElements expected.shape.toArray[Int])
    assert(outData.data().asFloat() sameElements expected.data)
  }

  it should "put dimensions in order in tensordot" in {
    val data: INDArray = Nd4j.create(
      Array(0f, 1f),
      Array(1, 2)
    )
    val matrix = Nd4j.create(
      Array(
        0f, 2f, 4f,
        1f, 3f, 5f
      ),
      Array(3, 2)
    )
    val out = Nd4jTensor.ops.tensordot(data, matrix, List((1, 1)), List((0, 0)), List((1, 0)))
    assert(out.shape() sameElements Array[Long](3, 1))
  }

  case class TParam[D: TensorData, S <: Shape](v: D, shape: S) extends Value[D, S] {

    override def field = implicitly[TensorField[D, S]]

    var grad: Option[D] = None

    override def dv(dv: D): Unit = {
      grad = grad.map { field.plus(_, dv) }.orElse(Some(dv))
    }

  }

}
