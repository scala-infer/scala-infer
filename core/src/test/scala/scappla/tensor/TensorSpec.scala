package scappla.tensor

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec
import scappla.Expr
import scappla.Functions.log

class TensorSpec extends FlatSpec {

  case class Batch(size: Int) extends Dim[Batch]

  case class Input(size: Int) extends Dim[Input]

  case class Other(size: Int) extends Dim[Other]

  it should "concatenate dimensions" in {
    val shape = Other(2) :#: Input(3) :#: Batch(2)

    val data = ArrayTensor(shape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f
    ))

    import TensorExpr._

    val batch = TensorExpr(shape, data)
    val logBatch = log(batch)
    print(s"Result: ${logBatch.v.data.data.mkString(",")}")
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(0.0f, 1.0f))

    val tensor = TensorExpr(shape, data)
    val sum = TensorExpr.sumAlong(tensor)
    val result = sum.v.data.data(0)

    print(s"Result $result")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(1f, 2f))

    var update: Option[ArrayTensor] = None

    val param = TensorExpr.param(
      Tensor(shape, data),
      (gradient: Tensor[Batch, ArrayTensor]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    TensorExpr.sumAlong(param)
        .dv(Tensor(Scalar, ArrayTensor(Seq.empty, Array(1f))))

    val dataArray = update.get
    assert(dataArray.data(0) == 2f)
    assert(dataArray.data(1) == 3f)
  }

  it should "buffer backward gradients" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(1f, 2f))
    val tensor = Tensor(shape, data)
    var update: Option[ArrayTensor] = None

    val param = TensorExpr.param(tensor,
      (gradient: Tensor[Batch, ArrayTensor]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    val buffer = param.buffer
    TensorExpr.sumAlong(buffer)
        .dv(Tensor(Scalar, ArrayTensor(Scalar.sizes, Array(1f))))

    assert(update.isEmpty)

    buffer.complete()

    val dataArray = update.get
    assert(dataArray.data(0) == 2f)
    assert(dataArray.data(1) == 3f)
  }

  it should "work with nd4j as well" in {
    implicit val nd4jTensor = Nd4jTensor.ops

    val shape = Batch(2)

    val data = Nd4j.create(
      Array(0.0f, 1.0f),
      shape.sizes.toArray
    )

    val tensor = TensorExpr(shape, data)
    val sum = TensorExpr.sumAlong(tensor)
    val result = sum.v.data.data().asFloat()(0)

    assert(result == 1f)
  }

  it should "einsum when multiplying" in {
    val inputDim = Input(2)
    val batchDim = Batch(3)
    val outDim = Other(2)
    val inputShape = inputDim :#: batchDim
    val outShape = outDim :#: batchDim

    val data = ArrayTensor(inputShape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f
    ))
    val matrix = ArrayTensor(outShape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f
    ))

    val input = TensorExpr(inputShape, data)
    val mTensor = TensorExpr(outShape, matrix)

    import TensorExpr._

    val out: Expr[Tensor[Input :#: Other, ArrayTensor]] = mTensor :*: input
    val outData = out.v.data
    val expected = ArrayTensor(Seq(2, 2), Array(5f, 14f, 14f, 50f))
    assert(outData.shape == expected.shape)
    assert(outData.data sameElements expected.data)
  }

}
