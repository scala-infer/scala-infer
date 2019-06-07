package scappla.tensor

import scala.collection.mutable
import scala.runtime.ScalaRunTime
import scala.util.Random

case class SparseTensor(shape: Seq[Int], data: Array[Float], coordinates: Seq[Array[Int]]) {

  override def equals(obj: Any): Boolean = {
    obj match {
      case st: SparseTensor =>
        st.shape == shape &&
            st.data.sameElements(data) &&
            st.coordinates.zip(coordinates)
                .forall {
                  case (l, r) => l.sameElements(r)
                }
      case _ => false
    }
  }

  override def toString: String = {
    def toStr[X](array: Array[X]): String = {
      s"Array(${array.mkString(", ")})"
    }
    s"SparseTensor($shape, ${toStr(data)}, ${coordinates.map { toStr }})"
  }
}

object SparseTensor {

  implicit val sparseIndexOrder: Ordering[Seq[Int]] = new Ordering[Seq[Int]] {

    def compare(x: Seq[Int], y: Seq[Int]): Int = {
      x.zip(y).foldLeft(0) { case (cur, (xi, yi)) =>
        cur match {
          case 0 => Integer.compare(xi, yi)
          case _ => cur
        }
      }
    }
  }

  implicit val sparseOps = new TensorData[SparseTensor] {

    def plus(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { _ + bF.getOrElse(0f) }.orElse(bF)
      })
    }

    def minus(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { _ - bF.getOrElse(0f) }.orElse(bF.map { -_ })
      })
    }

    def times(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.flatMap { af => bF.map { af * _ }}
      })
    }

    def div(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { af => af / bF.getOrElse(0f) }
      })
    }

    def pow(a: SparseTensor, b: SparseTensor): SparseTensor = {
      binary(a, b, { case (aF, bF) =>
        aF.map { af => bF.map { scala.math.pow(af, _).toFloat }.getOrElse(1f) }
      })
    }

    private def binary(
        a: SparseTensor,
        b: SparseTensor,
        fn: (Option[Float], Option[Float]) => Option[Float]
    ): SparseTensor = {
      assert(a.coordinates.size == b.coordinates.size)

      val aIter = a.data.zipWithIndex.iterator
      val bIter = b.data.zipWithIndex.iterator

      val resultIter = new Iterator[(Array[Int], Float)] {
        var aCursor: Option[(Float, Array[Int])] = None
        var bCursor: Option[(Float, Array[Int])] = None

        var out: Option[(Array[Int], Float)] = None

        def hasNext: Boolean = {
          step()
          out.isDefined
        }

        def next(): (Array[Int], Float) = {
          step()
          val result = out.get
          out = None
          result
        }

        private def step(): Unit = {
          while (!out.isDefined) {
            if (aCursor.isEmpty && aIter.hasNext) {
              val (aVal, aIndex) = aIter.next()
              aCursor = Some((aVal, a.coordinates.map { _(aIndex) }.toArray))
            }
            if (bCursor.isEmpty && bIter.hasNext) {
              val (bVal, bIndex) = bIter.next()
              bCursor = Some((bVal, b.coordinates.map { _(bIndex) }.toArray))
            }
            if (aCursor.isDefined && bCursor.isDefined) {
              val cmp = sparseIndexOrder.compare(aCursor.get._2, bCursor.get._2)
              if (cmp < 0) {
                out = fn(aCursor.map { _._1 }, None).map { (aCursor.get._2, _) }
                aCursor = None
              } else if (cmp > 0) {
                out = fn(None, bCursor.map { _._1 }).map { (bCursor.get._2, _) }
                bCursor = None
              } else {
                out = fn(aCursor.map{ _._1 }, bCursor.map{ _._1}).map { (aCursor.get._2, _) }
                aCursor = None
                bCursor = None
              }
            } else if (aCursor.isDefined) {
              out = fn(aCursor.map{ _._1 }, None).map { (aCursor.get._2, _) }
              aCursor = None
            } else if (bCursor.isDefined) {
              out = fn(None, bCursor.map{ _._1 }).map { (bCursor.get._2, _) }
              bCursor = None
            } else {
              return
            }
          }
        }
      }

      val zipped = resultIter.toArray
      SparseTensor(
        a.shape,
        zipped.map { _._2 },
        for { coordinate <- a.coordinates.indices } yield  {
          zipped.map { _._1(coordinate) }
        }
      )
    }

    def negate(a: SparseTensor): SparseTensor = {
      unary(a, -_)
    }

    def sqrt(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.sqrt(_).toFloat)
    }

    def log(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.log(_).toFloat)
    }

    def exp(a: SparseTensor): SparseTensor = {
      unary(a, scala.math.exp(_).toFloat)
    }

    def logistic(a: SparseTensor): SparseTensor = {
      unary(a, { x => 1f / (1f + scala.math.exp(-x).toFloat)})
    }

    def softplus(a: SparseTensor): SparseTensor = {
      unary(a, { x => scala.math.log1p(scala.math.exp(x)).toFloat})
    }

    private def unary(in: SparseTensor, fn: Float => Float): SparseTensor = {
      SparseTensor(in.shape, in.data.map(fn), in.coordinates)
    }

    def sumAll(a: SparseTensor): Float = {
      a.data.sum
    }

    // (de)constructing values

    def fill(value: Float, dims: Int*): SparseTensor = {
      SparseTensor(dims.toSeq, Array.fill(dims.product)(value), fullRank(dims))
    }

    def gaussian(dims: Int*): SparseTensor = {
      SparseTensor(dims.toSeq, Array.fill(dims.product)(Random.nextGaussian().toFloat), fullRank(dims))
    }

    private def fullRank(dims: Seq[Int]): Seq[Array[Int]] = {
      val totalSize = dims.product
      dims.scanLeft((1, totalSize)) { case ((prevSize, prevStride), dimSize) =>
        val size = prevSize * dimSize
        val stride = prevStride / dimSize
        (size, stride)
      }.zip(dims).map { case ((size, stride), dimSize) =>
        val result = Array.ofDim[Int](totalSize)
        for {
          i <- 0 until size
          j <- 0 until stride
        } {
          val index = i * stride + j
          result(index) = i % dimSize
        }
        result
      }
    }

    def count(d: SparseTensor, cond: Condition): Int = {
      cond match {
        case GreaterThan(value) if value > 0f =>
          d.data.count(_ > value)
        case GreaterThan(value) if value <= 0f =>
          d.shape.product - d.data.length + d.data.count(_ > value)
      }
    }

    def get(d: SparseTensor, indices: Int*): Float = {
      val (idx, idxp) = d.coordinates.zip(indices.toSeq).foldLeft((0, d.data.length)) {
        case ((start, end), (coordinates, toFind)) =>
          Range(start, end).foldLeft((end, end)) {
            case ((s, e), i) =>
              if (coordinates(i) < toFind) {
                (i + 1, i + 1)
              } else if (coordinates(i) == toFind) {
                (s, i + 1)
              } else {
                (s, e)
              }
          }
      }
      if (idxp == idx + 1) {
        d.data(idxp)
      } else {
        0f
      }
    }

    def put(d: SparseTensor, value: Float, indices: Int*): Unit = ???

    def imax(d: SparseTensor): Seq[Int] = {
      val index = d.data.zipWithIndex.maxBy(_._1)._2
      d.coordinates.map { _(index) }
    }

    def cumsum(a: SparseTensor, dim: Int): SparseTensor = ???

    // shape-affecting operations

    def sum(a: SparseTensor, dim: Int): SparseTensor = ???

    def broadcast(a: SparseTensor, dimIndex: Int, dimSize: Int): SparseTensor = ???

    def tensordot(a: SparseTensor, b: SparseTensor, ab: List[(Int, Int)], bc: List[(Int, Int)], ca: List[(Int, Int)]): SparseTensor = {
      import ScalaRunTime.stringOf

      val cShape: Seq[Int] = (bc.map {
        case (bi, ci) => (ci, b.shape(bi))
      } ++ ca.map {
        case (ci, ai) => (ci, a.shape(ai))
      }).sortBy { _._1 }
          .map { _._2 }

      def toDenseIndex(shape: Seq[Int], indices: Seq[Int]): Long = {
        indices.zip(shape).reverse.foldLeft(0L) { case (curIndex, (dimIndex, dimSize)) =>
            curIndex * dimSize + dimIndex
        }
      }
      def fromDenseIndex(shape: Seq[Int], linIndex: Long): Seq[Int] = {
        shape.scanLeft((linIndex, 0)) { case ((restLinIndex, _), dimSize) =>
          (restLinIndex / dimSize, (restLinIndex % dimSize).toInt)
        }.drop(1).map { _._2 }
      }

      val bByShared: Map[IndexedSeq[Int], Seq[(Array[Int], Float)]] = {
        // map of aIndex -> Seq[(cIndex, value)]
        val partitioned = mutable.HashMap[IndexedSeq[Int], Seq[(Array[Int], Float)]]()
            .withDefaultValue(Seq.empty)

        b.data.zipWithIndex.foreach { case (value, index) =>
          val bIndices = b.coordinates.map { _(index) }
          val aIndices = ab.map { case (_, bDim) =>
            bIndices(bDim)
          }.toVector
          val cIndices = bc.map { case (bDim, _) =>
            bIndices(bDim)
          }.toArray
          partitioned(aIndices) :+= (cIndices, value)
        }
        partitioned.toMap
      }

      val products: Seq[(Long, Float)] = a.data.toSeq.zipWithIndex.flatMap { case (aValue, aIndex) =>
        val aIndices = a.coordinates.map { _(aIndex) }
        val abIndices = ab.map { case (aDim, _) =>
          aIndices(aDim)
        }.toVector
        val acIndices: Seq[(Int, Int)] = ca.map { case (cDim, aDim) =>
          cDim -> aIndices(aDim)
        }
        for { (bcIndices, bValue) <- bByShared(abIndices) } yield {
          val fullBc = bcIndices.toSeq.zip(bc).map { case (idx, (_, cDim)) =>
            cDim -> idx
          }
          val cIndex = (acIndices ++ fullBc).sortBy { _._1 }.map { _._2 }
          toDenseIndex(cShape, cIndex) -> aValue * bValue
        }
      }

      val parts = products
        .sortBy { _._1 }
        .scanLeft(((-1L, 0f), None: Option[(Long, Float)])) {
          case (((prevIdx, prevVal), _), (idx, value)) =>
            if (idx == prevIdx) {
              ((idx, prevVal + value), None)
            } else {
              ((idx, value), Some((prevIdx, prevVal)))
            }
          }
      val out = parts
          .drop(2)
          .map { _._2 }
          .flatten
          .toList :+ parts.last._1

        /*
      val out = products
          .groupBy(_._1)
          .mapValues(_.map { _._2 }.sum)
          .toSeq
          .sortBy { _._1 }
          */
      val outIndices = out.map { case (i, _) => fromDenseIndex(cShape, i) }

      SparseTensor(
        cShape,
        out.map { _._2 }.toArray,
        cShape.indices.map { idx =>
          outIndices.map { _(idx) }.toArray
        }
      )
    }
  }

}
