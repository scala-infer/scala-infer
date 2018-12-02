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
    val result = sum.forward.getFloat(0)

    print(s"Result ${result}")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = Nd4j.create(Array(1f, 2f), Array(2))

    val param = TParam[Batch, INDArray](shape, nd4jTensor, () => data, {
      gradient => data.addi(gradient)
    })

    Tensor.sum(param)
        .backward(Nd4j.create(Array(1f), Array.empty[Int]))

    val dataArray = data.data().asFloat()
    assert(dataArray(0) == 2f)
    assert(dataArray(1) == 3f)
  }

}
