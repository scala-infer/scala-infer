package scappla.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  implicit val nd4jTensor = Nd4jTensor.ops

  it should "concatenate dimensions" in {

    val shape = Other(2) :#: Input(3) :#: Batch(2)

    val batch = Tensor(shape, Array(
      0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f
    ))
    val logBatch = log(batch)
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val data = Tensor(shape, Array(0.0f, 1.0f))
    val sum = Tensor.sum(data)
    val result = sum.forwardData.getFloat(0)

    print(s"Result ${result}")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = Tensor(shape, Array(1f, 2f))
    var update: Option[Array[Float]] = None

    val param = Tensor.param(data,
      (gradient: Tensor[Batch, INDArray]) => {
        val result = data.plus(gradient)
        update = Some(result.collect)
        result
      }
    )

    Tensor.sum(param).backward(Array(1f))

    val dataArray = update.get
    assert(dataArray(0) == 2f)
    assert(dataArray(1) == 3f)
  }

}
