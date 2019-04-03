package scappla.tensor

import org.scalatest.FlatSpec

class DataOpsSpec extends FlatSpec {

  it should "do einsum" in {
    val matrix = ArrayTensor(1 :: 2 :: Nil, Array(1.0f, 0.5f))
    val vector = ArrayTensor(2 :: 3 :: Nil, Array(1.0f, 0.6f, 0.3f, 0.4f, 0.5f, 2f))
    val product = DataOps.arrayOps.einsum(matrix, vector, List((1, 0)), List((1, 1)), List((0, 0)))
    assert(product.shape == Seq(1, 3))
  }

}
