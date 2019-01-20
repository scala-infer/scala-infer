package scappla.tensor

import org.scalatest.FlatSpec

class LenSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "calculate length of scalar" in {
    val result = Len[Scalar]
    assert(result() == 0)
  }

  it should "calculate length of single dim" in {
    val result = Len[Batch]
    assert(result() == 1)
  }

  it should "calculate length of product of dimensions" in {
    val result = Len[Other :#: Input :#: Batch]
    assert(result() == 3)
  }
}
