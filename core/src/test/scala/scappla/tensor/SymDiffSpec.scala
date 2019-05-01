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

  it should "recover indices" in {
 
    case class Height(size: Int) extends Dim[Height]

    case class Channel(size: Int) extends Dim[Channel]

    val lr: Aux[Height :#: Channel, Channel, Height] =
      SymDiff[Height :#: Channel, Channel]

    // Height :#: Channel  :*:  Channel  =>  Height

    // recoverLeft:
    // Channel  :*:  Height  =>  Height :#: Channel
    assert(lr.recoverLeft.matchedIndices == List())

    // recoverRight:
    // Height  :*:  Height :#: Channel  =>  Channel
    assert(lr.recoverRight.matchedIndices == List((0, 0)))
  }

  it should "backprop to original shape" in {

    case class Batch(size: Int) extends Dim[Batch]

    case class Input(size: Int) extends Dim[Input]

    case class Other(size: Int) extends Dim[Other]

    val inputDim = Input(1)
    val batchDim = Batch(2)
    val outDim = Other(3)
    val inputShape: Input :#: Batch = inputDim :#: batchDim
    assert(inputShape.sizes == List(1, 2))

    val outShape: Other :#: Batch = outDim :#: batchDim
    assert(outShape.sizes == List(3, 2))

    val sd: Aux[Input :#: Batch, Other :#: Batch, Input :#: Other] =
      SymDiff[Input :#: Batch, Other :#: Batch]
    val shape: Input :#: Other = sd.mapper.ab(inputShape, outShape)
    // println(s"LHS: ${inputShape}, RHS: ${outShape}, OUT: ${shape}")
    assert(shape.sizes == List(1, 3))
    assert(sd.matchedIndices == List((1, 1)))
    assert(sd.recoverLeft.matchedIndices == List((0, 1)))
    assert(sd.recoverRight.matchedIndices == List((0, 0)))

    val rrSd: Aux[Other :#: Batch, Input :#: Other, Input :#: Batch] = sd.recoverLeft
    val rShape: Input :#: Batch = rrSd.mapper.ab(outShape, shape)
    // println(s"LHS: ${outShape}, RHS: ${shape}, OUT: ${rShape}")
    assert(rShape.sizes == List(1, 2))
    assert(rrSd.matchedIndices == List((0, 1)))
    assert(rrSd.recoverLeft.matchedIndices == List((0, 0)))
    assert(rrSd.recoverRight.matchedIndices == List((1, 1)))

    val rlSd: Aux[Input :#: Other, Input :#: Batch, Other :#: Batch] = sd.recoverRight
    val lShape: Other :#: Batch = rlSd.mapper.ab(shape, inputShape)
    // println(s"LHS: ${shape}, RHS: ${inputShape}, OUT: ${lShape}")
    assert(lShape.sizes == List(3, 2))
    assert(rlSd.matchedIndices == List((0, 0)))
    assert(rlSd.recoverLeft.matchedIndices == List((1, 1)))
    assert(rlSd.recoverRight.matchedIndices == List((0, 1)))
  }

}
