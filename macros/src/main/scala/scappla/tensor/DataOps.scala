package scappla.tensor

import scala.util.Random

trait DataOps[D] {

  // (de)constructing values

  def fill(value: Float, dims: Int*): D

  def gaussian(shape: Int*): D

  // element-wise operations

  def plus(a: D, b: D): D

  def minus(a: D, b: D): D

  def times(a: D, b: D): D

  def div(a: D, b: D): D

  def negate(a: D): D

  def log(a: D): D

  def exp(a: D): D

  // shape-affecting operations

  def sumAll(a: D): Float

  def sum(a: D, dim: Int, shape: Int*): D

  def broadcast(a: D, dimIndex: Int, dimSize: Int, shape: Int*): D
}

object DataOps {

  implicit val arrayOps = new DataOps[Array[Float]] {

    override def fill(value: Float, shape: Int*): Array[Float] =
      Array.fill(shape.product)(value)

    override def gaussian(shape: Int*): Array[Float] = {
      Array.fill(shape.product)(Random.nextGaussian().toFloat)
    }

    override def plus(a: Array[Float], b: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = a(i) + b(i)
        i += 1
      }
      result
    }

    override def minus(a: Array[Float], b: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = a(i) - b(i)
        i += 1
      }
      result
    }

    override def times(a: Array[Float], b: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = a(i) * b(i)
        i += 1
      }
      result
    }

    override def div(a: Array[Float], b: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = a(i) / b(i)
        i += 1
      }
      result
    }

    override def negate(a: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = -a(i)
        i += 1
      }
      result
    }

    override def log(a: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.log(a(i)).toFloat
        i += 1
      }
      result
    }

    override def exp(a: Array[Float]): Array[Float] = {
      val len = a.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.exp(a(i)).toFloat
        i += 1
      }
      result
    }

    override def sum(a: Array[Float], dimIndex: Int, shape: Int*): Array[Float] = {
      val dimSize = shape(dimIndex)
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex - 1)
      val newShape = outerShape ++ innerShape

      val totalSize = newShape.product
      val innerSize = innerShape.product
      val output = Array.fill(totalSize)(0F)
      for {outer <- Range(0, outerShape.product)} {
        val baseOutputIdx = outer * innerSize
        for {i <- Range(0, innerSize)} {
          val outputIdx = i + baseOutputIdx
          var value = 0f
          for {d <- Range(0, dimSize)} {
            val inputIdx = i + d * innerSize + outer * innerSize * dimSize
            value += a(inputIdx)
          }
          output(outputIdx) += value
        }
      }
      output
    }

    override def broadcast(a: Array[Float], dimIndex: Int, dimSize: Int, shape: Int*): Array[Float] = {
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex)
      val newShape = (outerShape :+ dimSize) ++ innerShape
      val output = Array.ofDim[Float](newShape.product)

      val innerSize = innerShape.product
      for {outer <- Range(0, outerShape.product)} {
        val baseInput = outer * innerSize
        for {i <- Range(0, innerSize)} {
          val inputIdx = i + baseInput
          val value = a(inputIdx)
          for {d <- Range(0, dimSize)} {
            val outputIdx = i + d * innerSize + outer * innerSize * dimSize
            output(outputIdx) = value
          }
        }
      }
      output
    }

    override def sumAll(a: Array[Float]): Float = {
      val len = a.length
      var result = 0f
      var i = 0
      while (i < len) {
        result += a(i)
        i += 1
      }
      result
    }
  }
}
