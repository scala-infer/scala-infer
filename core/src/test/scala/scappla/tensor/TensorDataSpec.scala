package scappla.tensor

import org.scalatest.FlatSpec

class TensorDataSpec extends FlatSpec {

  val ops = implicitly[TensorData[ArrayTensor]]

  it should "do tensordot" in {
    val matrix = ArrayTensor(1 :: 2 :: Nil, Array(
      1.0f, 0.5f,
    ))
    val vector = ArrayTensor(2 :: 3 :: Nil, Array(
      1.0f, 0.6f, 0.3f,
      0.4f, 0.5f, 2f,
    ))
    val product = ops.tensordot(matrix, vector, List((1, 0)), List((1, 1)), List((0, 0)))
    assert(product.shape == Seq(1, 3))
    assert(product.data sameElements Array(
      1.2f, 0.85f, 1.3f,
    ))
  }

  it should "find imax" in {
    val data = ArrayTensor(Seq(2, 3), Array(
      0f, 1f, 2f,
      5f, 4f, 3f,
    ))
    val m = ops.imax(data)
    assert(m == Seq(1, 0))
    assert(ops.get(data, m: _*) == 5f)
  }

  it should "cumsum" in {
    val data = ArrayTensor(Seq(2, 3), Array(
      0f, 1f, 2f,
      3f, 4f, 5f,
    ))
    val byrow = ops.cumsum(data, 0)
    assert(byrow.data sameElements Array(
      0f, 1f, 2f,
      3f, 5f, 7f,
    ))
    val bycol = ops.cumsum(data, 1)
    assert(bycol.data sameElements Array(
      0f, 1f, 3f,
      3f, 7f, 12f,
    ))
  }

  it should "sum" in {
    val data = ArrayTensor(Seq(2, 3), Array(
      0f, 1f, 2f,
      3f, 4f, 5f,
    ))
    val byrow = ops.sum(data, 0)
    assert(byrow.data sameElements Array(
      3f, 5f, 7f
    ))
    val bycol = ops.sum(data, 1)
    assert(bycol.data sameElements Array(
      3f,
      12f
    ))
  }

  it should "count" in {
    val data = ArrayTensor(Seq(2, 3), Array(
      0f, 1f, 2f,
      3f, 4f, 5f,
    ))
    val cnt = ops.count(data, GreaterThan(3.5f))
    assert(cnt == 2)
  }

  it should "broadcast" in {
    val data = ArrayTensor(Seq(2, 3), Array(
      0f, 1f, 2f,
      3f, 4f, 5f,
    ))
    val bc = ops.broadcast(data, 1, 2)
    assert(bc.shape == Seq(2, 2, 3))
    assert(bc.data sameElements Array(
      0f, 1f, 2f,
      0f, 1f, 2f,
      3f, 4f, 5f,
      3f, 4f, 5f,
    ))
  }

}
