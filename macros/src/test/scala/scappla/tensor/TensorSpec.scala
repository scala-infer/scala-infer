package scappla.tensor

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  it should "concatenate dimensions" in {
    case class Batch(size: Int) extends Dim[Batch]

    case class Input(size: Int) extends Dim[Input]

    case class Other(size: Int) extends Dim[Other]

    val shape = Other(4) :: Input(3) :: Batch(5)

    val batch: Tensor[Other :: Input :: Batch] = TConst(
      Nd4j.create(Array(0.0, 1.0, 2.0, 3.0, 4.0)),
      Other(2) :: Input(3) :: Batch(5)
    )

    val logBatch = log(batch)
  }
}
