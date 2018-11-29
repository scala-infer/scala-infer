package scappla.tensor

import org.scalatest.FlatSpec
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "concatenate dimensions" in {

    val shape = Other(2) :#: Input(3) :#: Batch(2)

    val batch: Tensor[Other :#: Input :#: Batch] = TParam(shape, _ => ())
    val logBatch = log(batch)
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val batch: Tensor[Batch] = TParam(shape, _ => ())

    val result = Nd4jTensor.interpret(Tensor.sum(batch), param => Array(0.0f, 1.0f))

    print(s"Result ${result.v}")
  }

}
