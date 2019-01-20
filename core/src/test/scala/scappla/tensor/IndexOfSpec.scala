package scappla.tensor

import org.scalatest.FlatSpec

class IndexOfSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  it should "calculate index with single element" in {
    assert(IndexOf[Input, Input].toInt == 0)
  }

  it should "calculate index in sequence" in {
    assert(IndexOf[Input :#: Batch, Input].toInt == 0)
    assert(IndexOf[Input :#: Batch, Batch].toInt == 1)
  }
}
