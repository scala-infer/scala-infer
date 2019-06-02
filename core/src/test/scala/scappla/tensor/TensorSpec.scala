package scappla.tensor

import org.scalatest.FlatSpec
import scappla._
import Tensor._
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

    val batch = Value(data, shape)
    val logBatch = log(batch)
    print(s"Result: ${logBatch.v.data.mkString(",")}")
  }

  it should "sum along dimension" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(0.0f, 1.0f))

    val tensor = Constant(data, shape)
    val sum = sumAlong(tensor, shape)
    val result = sum.v.data(0)

    print(s"Result $result")
  }

  it should "backprop gradient" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(1f, 2f))

    val param = TParam(data, shape)

    sumAlong(param, shape)
        .dv(ArrayTensor(Seq.empty, Array(1f)))

    val update = param.grad
    val dataArray = update.get
    assert(dataArray.data(0) == 1f)
    assert(dataArray.data(1) == 1f)
  }

  it should "buffer backward gradients" in {
    val shape = Batch(2)

    val data = ArrayTensor(shape.sizes, Array(1f, 2f))

    val param = TParam(data, shape)

    val buffer = param.buffer
    sumAlong(buffer, shape)
        .dv(ArrayTensor(Scalar.sizes, Array(1f)))

    var update = param.grad
    assert(update.isEmpty)

    buffer.complete()

    update = param.grad
    val dataArray = update.get
    assert(dataArray.data(0) == 1f)
    assert(dataArray.data(1) == 1f)
  }

  it should "tensordot when multiplying" in {
    val inputDim = Input(1)
    val batchDim = Batch(2)
    val outDim = Other(3)
    val inputShape = inputDim :#: batchDim
    val outShape = outDim :#: batchDim

    val data = ArrayTensor(inputShape.sizes, Array(
      0f, 1f
    ))
    val input = Value(data, inputShape)

    val matrix = ArrayTensor(outShape.sizes, Array(
      0f, 1f,
      2f, 3f,
      4f, 5f
    ))
    val param = TParam(matrix, outShape)

    val out: Value[ArrayTensor, Input :#: Other] = param :*: input
    val outData = out.v
    val expected = ArrayTensor(Seq(1, 3), Array(1f, 3f, 5f))
    assert(outData.shape == expected.shape)
    assert(outData.data sameElements expected.data)

    val grad = ArrayTensor(out.shape.sizes, Array(1f, 1f, 1f))
    out.dv(grad)

    val dataArray = param.grad.get
    println(s"GRAD: ${dataArray.data.mkString(", ")}")
    assert(dataArray.data sameElements Array(0f, 1f, 0f, 1f, 0f, 1f))
  }

  it should "find imax" in {
    val inputDim = Input(2)
    val batchDim = Batch(3)
    type Shape = Input :#: Batch
    val shape = inputDim :#: batchDim

    val data = ArrayTensor(shape.sizes, Array(
      0f, 1f, 2f, 3f, 4f, 5f
    ))
    val input = Value(data, shape)

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

    val x = Value(xData, xShape)
    val y = Value(yData, yShape)

    val z: Value[ArrayTensor, C :#: A] = x :*: y
    z.dv(ArrayTensor((c :#: a).sizes, Array(1f, 2f, 3f)))
  }

  case class TParam[D: TensorData, S <: Shape](v: D, shape: S) extends Value[D, S] {

    override def field = implicitly[TensorField[D, S]]

    var grad: Option[D] = None

    override def dv(dv: D): Unit = {
      grad = grad.map { field.plus(_, dv) }.orElse(Some(dv))
    }

  }

}
