package scappla.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "concatenate dimensions" in {
    val shape = Other(2) :#: Input(3) :#: Batch(2)

    val data = Array(
      0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f
    )

    val batch = TensorExpr(shape, data)
    val logBatch = log(batch)
    print(s"Result: ${logBatch.v.data.mkString(",")}")
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val data = Array(0.0f, 1.0f)

    val tensor = TensorExpr(shape, data)
    val sum = TensorExpr.sumAlong(tensor)
    val result = sum.v.data(0)

    print(s"Result $result")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = Array(1f, 2f)

    val tensor = TensorExpr(shape, data)
    var update: Option[Array[Float]] = None

    val param = TensorExpr.param(
      Tensor(shape, data),
      (gradient: Tensor[Batch, Array[Float]]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    TensorExpr.sumAlong(param)
        .dv(Tensor(Scalar, Array(1f)))

    val dataArray = update.get
    assert(dataArray(0) == 2f)
    assert(dataArray(1) == 3f)
  }

  it should "buffer backward gradients" in {
    val shape = Batch(2)

    val data = Array(1f, 2f)
    val tensor = Tensor(shape, data)
    var update: Option[Array[Float]] = None

    val param = TensorExpr.param(tensor,
      (gradient: Tensor[Batch, Array[Float]]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    val buffer = param.buffer
    TensorExpr.sumAlong(buffer)
        .dv(Tensor(Scalar, Array(1f)))

    assert(update.isEmpty)

    buffer.complete()

    val dataArray = update.get
    assert(dataArray(0) == 2f)
    assert(dataArray(1) == 3f)
  }

  it should "work with nd4j as well" in {
    implicit val nd4jTensor = Nd4jTensor.ops

    val shape = Batch(2)

    val data = Nd4j.create(
      Array(0.0f, 1.0f),
      shape.sizes.toArray
    )

    val tensor = TensorExpr(shape, data)
    val sum = TensorExpr.sumAlong(tensor)
    val result = sum.v.data.data().asFloat()(0)

    assert(result == 1f)
  }

}
