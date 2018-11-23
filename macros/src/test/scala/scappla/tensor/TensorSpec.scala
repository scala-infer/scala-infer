package scappla.tensor

import org.scalatest.FlatSpec
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "concatenate dimensions" in {

    val shape = Other(2) :#: Input(3) :#: Batch(2)

    val batch: Tensor[Other :#: Input :#: Batch] = TConst(
      Array(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f),
      shape
    )

    val logBatch = log(batch)
  }

}
