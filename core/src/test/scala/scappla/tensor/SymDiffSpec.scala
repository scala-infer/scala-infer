package scappla.tensor

import org.scalatest.FlatSpec
import scappla.tensor.SymDiff.Aux

class SymDiffSpec extends FlatSpec {

  case class One(size: Int) extends Dim[One]

  case class Two(size: Int) extends Dim[Two]

  case class Three(size: Int) extends Dim[Three]

  it should "not find any indices in a scalar" in {
    val sd = SymDiff[One, Scalar]
    assert(sd.matchedIndices.isEmpty)
  }

  it should "not find any index in a dim" in {
    val sd = SymDiff[One, Two]
    assert(sd.matchedIndices.isEmpty)
  }

  it should "collapse identical dims" in {
    val sd = SymDiff[One, One]
    assert(sd.matchedIndices.size == 1)

    val m = sd.matchedIndices.head
    assert(m._1 == 0)
    assert(m._2 == 0)
  }

  it should "not find any index in a shape" in {
    val sd = SymDiff[One :#: Two, Three]
    assert(sd.matchedIndices.isEmpty)
  }

  it should "find leading indices" in {
    val sd = SymDiff[One, One :#: Two]
    assert(sd.matchedIndices.size == 1)

    val m = sd.matchedIndices.head
    assert(m._1 == 0)
    assert(m._2 == 0)
  }

  it should "find non-leading indices" in {
    val sd = SymDiff[One, Two :#: One]
    assert(sd.matchedIndices.size == 1)

    val m = sd.matchedIndices.head
    assert(m._1 == 0)
    assert(m._2 == 1)
  }

  it should "find corresponding indices in shapes" in {
    val sd = SymDiff[One :#: Two, Two :#: Three]
    assert(sd.matchedIndices.size == 1)

    val m = sd.matchedIndices.head
    assert(m._1 == 1)
    assert(m._2 == 0)
  }

  it should "not find corresponding leading indices in shapes" in {
    val sd = SymDiff[One :#: Two, Three :#: One]
    assert(sd.matchedIndices.size == 1)

    val m = sd.matchedIndices.head
    assert(m._1 == 0)
    assert(m._2 == 1)
  }

  it should "collapse equivalent shapes" in {
    val sd = SymDiff[One :#: Two, Two :#: One]
    assert(sd.matchedIndices.size == 2)

    val one = sd.matchedIndices(0)
    assert(one._1 == 0)
    assert(one._2 == 1)

    val two = sd.matchedIndices(1)
    assert(two._1 == 1)
    assert(two._2 == 0)
  }

  it should "find nested contraction" in {

    case class Height(size: Int) extends Dim[Height]

    case class Channel(size: Int) extends Dim[Channel]

    val lr: Aux[Height :#: Channel, Channel, Height] =
      SymDiff[Height :#: Channel, Channel]

    assert(lr.matchedIndices == List((1, 0)))
  }

}
