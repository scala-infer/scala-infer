package scappla.tensor

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import scappla.Real

import scala.collection.mutable

object Nd4jTensor extends TensorInterpreter {

  override def interpret(tensor: Tensor[Scalar], resolver: TParam[_] => Array[Float]): Real = {
    new Nd4jReal(tensor, resolver)
  }

  private class Nd4jReal(tensor: Tensor[Scalar], resolver: TParam[_] => Array[Float]) extends Real {

    private val cache: mutable.Map[Tensor[_], INDArray] =
      mutable.HashMap.empty[Tensor[_], INDArray]

    override def v: Double = {
      forwardShape(tensor).getDouble(Seq(0): _*)
    }

    private def forwardShape[S <: Shape](variable: Tensor[S]): INDArray = {
      if (!cache.contains(variable)) {
        val result = variable match {
          case param@TParam(shape, _) =>
            Nd4j.create(resolver(param), shape.sizes.toArray)
          case TNeg(orig) =>
            forwardShape(orig).neg()
          case TPlus(left, right) =>
            forwardShape(left).add(forwardShape(right))
          case TMinus(left, right) =>
            forwardShape(left).sub(forwardShape(right))
          case TTimes(left, right) =>
            forwardShape(left).mul(forwardShape(right))
          case TDiv(left, right) =>
            forwardShape(left).div(forwardShape(right))
          case TLog(upstream) =>
            Transforms.log(forwardShape(upstream))
          case TExp(upstream) =>
            Transforms.exp(forwardShape(upstream))
          case TSum(shape, index, upstream) =>
            forwardShape(upstream).sum(index)
        }
        cache += variable -> result
      }
      cache(variable)
    }

    override def dv(value: Double): Unit = {
      backwardShape(tensor, Nd4j.create(Array(value), Array.empty[Int]))
    }

    private def backwardShape[S <: Shape](tensor: Tensor[S], gradient: INDArray) {
      tensor match {
        case param@TParam(shape, backward) =>
          backward(gradient.data().getFloatsAt(0L, shape.size))
        case TNeg(orig) =>
          backwardShape(orig, gradient.neg())
        case TPlus(left, right) =>
          backwardShape(left, gradient)
          backwardShape(right, gradient)
        case TMinus(left, right) =>
          backwardShape(left, gradient)
          backwardShape(right, gradient.neg())
        case TTimes(left, right) =>
          backwardShape(left, gradient.mul(forwardShape(right)))
          backwardShape(right, gradient.mul(forwardShape(left)))
        case TDiv(numer, denom) =>
          val numer_v = forwardShape(numer)
          val denom_v = forwardShape(denom)
          backwardShape(numer, gradient.div(denom_v))
          backwardShape(denom, gradient
              .neg().muli(numer_v).divi(denom_v).divi(denom_v)
          )
        case TLog(upstream) =>
          backwardShape(upstream, gradient.div(forwardShape(upstream)))
        case TExp(upstream) =>
          backwardShape(upstream, gradient.mul(forwardShape(upstream)))
        case TSum(shape, index, upstream) =>
          backwardShape(upstream, gradient.broadcast())
      }
    }
  }

}

