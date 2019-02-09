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

  def pow(a: D, b: D): D

  def negate(a: D): D

  def sqrt(a: D): D

  def log(a: D): D

  def exp(a: D): D

  // shape-affecting operations

  def sumAll(a: D): Float

  def sum(a: D, dim: Int): D

  def broadcast(a: D, dimIndex: Int, dimSize: Int): D

  def einsum(a: D, b: D, dims: (Int, Int)*): D
}

case class ArrayTensor(shape: Seq[Int], data: Array[Float])

object DataOps {

  implicit val arrayOps = new DataOps[ArrayTensor] {

    override def fill(value: Float, shape: Int*): ArrayTensor =
      ArrayTensor(shape.toSeq, Array.fill(shape.product)(value))

    override def gaussian(shape: Int*): ArrayTensor = {
      ArrayTensor(shape.toSeq, Array.fill(shape.product)(Random.nextGaussian().toFloat))
    }

    override def plus(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) + bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def minus(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) - bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def times(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) * bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def div(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = ad(i) / bd(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def pow(a: ArrayTensor, b: ArrayTensor): ArrayTensor = {
      assert(a.shape == b.shape)
      val (ad, bd) = (a.data, b.data)
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.pow(ad(i), bd(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def negate(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = -ad(i)
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def sqrt(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.sqrt(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def log(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.log(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def exp(a: ArrayTensor): ArrayTensor = {
      val ad = a.data
      val len = ad.length
      val result = new Array[Float](len)
      var i = 0
      while (i < len) {
        result(i) = scala.math.exp(ad(i)).toFloat
        i += 1
      }
      ArrayTensor(a.shape, result)
    }

    override def sum(a: ArrayTensor, dimIndex: Int): ArrayTensor = {
      val (shape, data) = (a.shape, a.data)
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
            value += data(inputIdx)
          }
          output(outputIdx) += value
        }
      }
      ArrayTensor(newShape, output)
    }

    override def broadcast(a: ArrayTensor, dimIndex: Int, dimSize: Int): ArrayTensor = {
      val (shape, data) = (a.shape, a.data)
      val outerShape = shape.take(dimIndex)
      val innerShape = shape.takeRight(shape.size - dimIndex)
      val newShape = (outerShape :+ dimSize) ++ innerShape
      val output = Array.ofDim[Float](newShape.product)

      val innerSize = innerShape.product
      for {outer <- Range(0, outerShape.product)} {
        val baseInput = outer * innerSize
        for {i <- Range(0, innerSize)} {
          val inputIdx = i + baseInput
          val value = data(inputIdx)
          for {d <- Range(0, dimSize)} {
            val outputIdx = i + d * innerSize + outer * innerSize * dimSize
            output(outputIdx) = value
          }
        }
      }
      ArrayTensor(newShape, output)
    }

    override def sumAll(a: ArrayTensor): Float = {
      val data = a.data
      val len = data.length
      var result = 0f
      var i = 0
      while (i < len) {
        result += data(i)
        i += 1
      }
      result
    }

    override def einsum(a: ArrayTensor, b: ArrayTensor, dims: (Int, Int)*): ArrayTensor = {
      val aContract = dims.map {
        _._1
      }.toSet
      val bContract = dims.map {
        _._2
      }.toSet
      val aRemnants = a.shape.zipWithIndex.filterNot { ai => aContract.contains(ai._2) }
      val bRemnants = b.shape.zipWithIndex.filterNot { bi => bContract.contains(bi._2) }
      val newShape = aRemnants.map {
        _._1
      } ++ bRemnants.map {
        _._1
      }

      val aDims = a.shape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap

      val bDims = b.shape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap

      val cDims = newShape.scanRight(1) {
        case (dimSize, stride) => dimSize * stride
      }.drop(1).zipWithIndex.map {
        _.swap
      }.toMap

      val abDimensions = dims.map {
        case (ai, bi) =>
          ((aDims(ai), bDims(bi)), a.shape(ai))
      }
      val caDimensions = aRemnants.zipWithIndex.map {
        case ((_, ai), ci) =>
          ((cDims(ci), aDims(ai)), a.shape(ai))
      }
      val bcDimensions = bRemnants.zipWithIndex.map {
        case ((_, bi), ci) =>
          ((bDims(bi), cDims(ci + aRemnants.size)), b.shape(bi))
      }

      // take pairs of matching strides and dimension size, generate indices
      def nestedIter(parts: Seq[((Int, Int), Int)]): Iterator[(Int, Int)] = {
        parts.foldLeft(Iterator.single((0, 0))) {
          case (current, ((strideLeft, strideRight), dimSize)) =>
            current.flatMap { case (indexLeft, indexRight) =>
              (0 until dimSize).iterator.map { i =>
                (indexLeft + i * strideLeft, indexRight + i * strideRight)
              }
            }
        }
      }

      val result = Array.ofDim[Float](newShape.product)
      for { (cBaseIndex, aBaseIndex) <- nestedIter(caDimensions)} {
        for { (aRelIndex, bBaseIndex) <- nestedIter(abDimensions)} {
          val aIndex: Int = aBaseIndex + aRelIndex

          val aValue = a.data(aIndex)
          for {(bRelIndex, cRelIndex) <- nestedIter(bcDimensions)} {
            val bIndex: Int = bBaseIndex + bRelIndex
            val cIndex: Int = cBaseIndex + cRelIndex

            val bValue = b.data(bIndex)
            result(cIndex) += aValue * bValue
          }
        }
      }

      ArrayTensor(newShape, result)
    }
  }

}
