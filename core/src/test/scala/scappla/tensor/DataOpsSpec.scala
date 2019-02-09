package scappla.tensor

import org.scalatest.FlatSpec

class DataOpsSpec extends FlatSpec {

  it should "do einsum" in {
    val matrix = ArrayTensor(2 :: 2 :: Nil, Array(1.0f, 0.5f, 0.5f, 0.5f))
    val vector = ArrayTensor(2 :: Nil, Array(1.0f, 0.6f))
    val product = DataOps.arrayOps.einsum(matrix, vector, (1, 0))
    assert(product.shape == Seq(2))
    assert(product.data sameElements Array(1.3f, 0.8f))
  }

}
