package scappla.tensor

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FlatSpec
import scappla.Value
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

    import TensorValue._

    val batch = TensorValue(shape, data)
    val logBatch = log(batch)
    print(s"Result: ${logBatch.v.data.data.mkString(",")}")
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(0.0f, 1.0f))

    val tensor = TensorValue(shape, data)
    val sum = TensorValue.sumAlong(tensor, shape)
    val result = sum.v.data.data(0)

    print(s"Result $result")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(1f, 2f))

    var update: Option[ArrayTensor] = None

    val param = TensorValue.param(
      Tensor(shape, data),
      (gradient: Tensor[Batch, ArrayTensor]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    TensorValue.sumAlong(param, shape)
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

    val param = TensorValue.param(tensor,
      (gradient: Tensor[Batch, ArrayTensor]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(shape, result)
      }
    )

    val buffer = param.buffer
    TensorValue.sumAlong(buffer, shape)
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

    val tensor = TensorValue(shape, data)
    val sum = TensorValue.sumAlong(tensor, shape)
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
    val input = TensorValue(inputShape, data)

    var update: Option[ArrayTensor] = None
    val matrix = ArrayTensor(outShape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f
    ))
    val param = TensorValue.param(Tensor(outShape, matrix),
      (gradient: Tensor[Other :#: Batch, ArrayTensor]) => {
        val result = DataOps.arrayOps.plus(data, gradient.data)
        update = Some(result)
        Tensor(outShape, result)
      }
    )

    import TensorValue._

    val out: Value[Tensor[Input :#: Other, ArrayTensor]] = param :*: input
    val outData = out.v.data
    val expected = ArrayTensor(Seq(2, 2), Array(5f, 14f, 14f, 50f))
    assert(outData.shape == expected.shape)
    assert(outData.data sameElements expected.data)

    val grad = Tensor(
      out.v.shape,
      ArrayTensor(out.v.shape.sizes, Array(1f, 1f, 1f, 1f))
    )
    out.dv(grad)

    val dataArray = update.get
    assert(dataArray.data sameElements Array(3f, 6f, 9f, 6f, 9f, 12f))
  }

  it should "find imax" in {
    val inputDim = Input(2)
    val batchDim = Batch(3)
    type Shape = Input :#: Batch
    val shape = inputDim :#: batchDim

    val data = ArrayTensor(shape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f
    ))
    val input = TensorValue(shape, data)

    import TensorValue._

    val index = maxIndex(input)
    assert(index == Index[Shape](List(1, 2)))
  }

  it should "order indices correctly" in {

    case class A(size: Int) extends Dim[A]
    case class B(size: Int) extends Dim[B]
    case class C(size: Int) extends Dim[C]

    val a = A(1)
    val b = B(2)
    val c = C(3)

    val xShape = a :#: b
    val xData = ArrayTensor(xShape.sizes, Array(1f, 2f))

    val yShape = b :#: c
    val yData = ArrayTensor(yShape.sizes, Array(1f, 2f, 3f, 4f, 5f, 6f))

    val x = TensorValue(xShape, xData)
    val y = TensorValue(yShape, yData)

    import TensorValue._

    val z: Value[Tensor[C :#: A, ArrayTensor]] = x :*: y
    z.dv(Tensor.apply(c :#: a, ArrayTensor((c :#: a).sizes, Array(1f, 2f, 3f))))
  }

}
