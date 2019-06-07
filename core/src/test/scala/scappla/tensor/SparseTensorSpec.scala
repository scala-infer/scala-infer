package scappla.tensor

import scala.util.Random

import org.scalatest.FlatSpec

class SparseTensorSpec extends FlatSpec {

  val ops = implicitly[TensorData[SparseTensor]]

  it should "do elemwise ops" in {

    val a = SparseTensor(
      Seq(
        2,
        3
      ),
      Array(1f, 2f),
      Seq(
        Array(0, 0),
        Array(1, 2)
      )
    )
    val b = SparseTensor(
      Seq(
        2,
        3
      ),
      Array(3f, 4f),
      Seq(
        Array(0, 1),
        Array(1, 1)
      )
    )

    {
      val c = ops.plus(a, b)
      assert(c == SparseTensor(
        Seq(
          2,
          3
        ),
        Array(4f, 2f, 4f),
        Seq(
          Array(0, 0, 1),
          Array(1, 2, 1)
        )
      ))
    }

    {
      val c = ops.minus(a, b)
      assert(c == SparseTensor(
        Seq(
          2,
          3
        ),
        Array(-2f, 2f, -4f),
        Seq(
          Array(0, 0, 1),
          Array(1, 2, 1)
        )
      ))
    }

    {
      val c = ops.times(a, b)
      assert(c == SparseTensor(
        Seq(
          2,
          3
        ),
        Array(3f),
        Seq(
          Array(0),
          Array(1)
        )
      ))
    }
  }

  it should "do tensordot" in {

    val matrix = SparseTensor(1 :: 2 :: Nil,
      Array(1.0f, 0.5f),
      Seq(
        Array(0, 0),
        Array(0, 1)
      )
    )
    val vector = SparseTensor(2 :: 3 :: Nil,
      Array(
        1.0f, 0.6f, 0.3f,
        0.4f, 0.5f, 2f,
      ),
      Seq(
        Array(0, 0, 0, 1, 1, 1),
        Array(0, 1, 2, 0, 1, 2)
      )
    )
    val product = ops.tensordot(matrix, vector, List((1, 0)), List((1, 1)), List((0, 0)))
    assert(product == SparseTensor(
      Seq(1, 3),
      Array(1.2f, 0.85f, 1.3f),
      Seq(
        Array(0, 0, 0),
        Array(0, 1, 2)
      )
    ))
  }

  it should "do fast sparse tensordot" in {
    val N = 100000
    val K = 100
    val matrix = SparseTensor(N :: K :: Nil,
      Array.fill(N)(Random.nextGaussian().toFloat),
      Seq(
        Range(0, N).toArray,
        Array.fill(N)(Random.nextInt(K))
      )
    )
    val vector = SparseTensor(K :: Nil,
      Array.fill(K)(Random.nextGaussian().toFloat),
      Seq(
        Range(0, K).toArray,
      )
    )

    // warm up
    ops.tensordot(matrix, vector, List((1, 0)), List.empty, List((0, 0)))

    val start = System.currentTimeMillis()
    val product = ops.tensordot(matrix, vector, List((1, 0)), List.empty, List((0, 0)))
    val end = System.currentTimeMillis()
    println(s"Sparse mv time: ${(1000000 * (end - start)) / N} ns / row")
  }

}
