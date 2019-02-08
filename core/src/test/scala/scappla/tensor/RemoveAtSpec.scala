package scappla.tensor

import org.scalatest.FlatSpec
import shapeless._

class RemoveAtSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "remove only index to resolve to scalar" in {
    val in = Batch(3)
    val r = RemoveAt[Batch, _0]
    assert(r(in) == Scalar)
  }

  it should "remove leading index from sequence" in {
    val in = Batch(3) :#: Input(2)
    val r = RemoveAt[Batch :#: Input, _0]
    assert(r(in) == Input(2))
  }

  it should "remove other non-leading index" in {
    val in = Batch(3) :#: Input(2)
    val r = RemoveAt[Batch :#: Input, Succ[_0]]
    assert(r(in) == Batch(3))
  }

  it should "remove middle index" in {
    val in = Batch(3) :#: Input(2) :#: Other(4)
    val r = RemoveAt[Batch :#: Input :#: Other, Succ[_0]]
    assert(r(in) == Batch(3) :#: Other(4))
  }

}
