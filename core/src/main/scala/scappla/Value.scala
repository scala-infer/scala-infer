package scappla

import scappla.tensor.ArrayTensor

import scala.util.Random

trait Value[X, S] {

  def field: BaseField[X, S]

  def shape: S

  def v: X

  def dv(v: X): Unit

  def unary_-(): Value[X, S] =
    VNegate(this)

  def +(other: Value[X, S]): Value[X, S] =
    VPlus(this, other)

  def -(other: Value[X, S]): Value[X, S] =
    VMinus(this, other)

  def *(other: Value[X, S]): Value[X, S] =
    VTimes(this, other)

  def /(other: Value[X, S]): Value[X, S] =
    VDiv(this, other)

  def buffer: Buffered[X, S] =
    VBuffer(this)

  def const: Value[X, S] =
    Constant(this.v, this.shape)(field)
}

object Value {

  def apply[X, S](value: X, shape: S)(implicit bf: BaseField[X, S]): Value[X, S] = Constant(value, shape)

  implicit def apply(value: Double): Value[Double, Unit] = Constant(value, ())
}

trait Buffered[X, S] extends Value[X, S] with Completeable

class Constant[X, S](val v: X, val shape: S, val field: BaseField[X, S]) extends Buffered[X, S] {

  override def dv(v: X): Unit = {}

  override def buffer: Buffered[X, S] = this

  override def complete(): Unit = {}

  override def toString: String = {
    s"Cst($v)"
  }
}

object Constant {

  def apply[X, S](v: X, shape: S)(implicit bf: BaseField[X, S]): Constant[X, S] = new Constant(v, shape, bf)
}

trait BaseField[X, S] extends Elemwise[X] {

  def fromInt(x: Int, shape: S): X

  def fromDouble(x: Double, shape: S): X

  def gaussian(shape: S): X
}

object BaseField {

  implicit val doubleBaseField: BaseField[Double, Unit] =
    new BaseField[Double, Unit] {

      override def fromInt(x: Int, shape: Unit): Double = x.toDouble

      override def fromDouble(x: Double, shape: Unit): Double = x

      override def gaussian(shape: Unit): Double = Random.nextGaussian()

      override def negate(x: Double): Double = -x

      override def plus(x: Double, y: Double): Double = x + y

      override def minus(x: Double, y: Double): Double = x - y

      override def times(x: Double, y: Double): Double = x * y

      override def div(x: Double, y: Double): Double = x / y

      override def sqrt(x: Double): Double = math.sqrt(x)

      override def log(x: Double): Double = math.log(x)

      override def exp(x: Double): Double = math.exp(x)

      override def pow(x: Double, y: Double): Double = math.pow(x, y)

      override def logistic(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

      override def softplus(x: Double): Double = math.log1p(math.exp(x))

      override def sumAll(x: Double): Float = x.toFloat
    }

}

case class VBuffer[D, S](upstream: Value[D, S])
    extends Value[D, S] with Buffered[D, S] {

  private var refCount: Int = 1

  private var grad: Option[D] = None

  override val field = upstream.field

  override val shape: S = upstream.shape

  override val v: D = upstream.v

//  println(s"  BUFFERING  $this")

  override def dv(gradient: D): Unit = {
    grad = grad.map {
      field.plus(_, gradient)
    }.orElse(Some(gradient))
    /*
    grad.get match {
      case tensor: ArrayTensor =>
        val finite = tensor.data.forall { f =>
            !f.isNaN && !f.isInfinite
          }
        if (!finite) {
          assert(false)
        }
      case f: Double =>
        if (f.isNaN || f.isInfinite) {
          assert(false)
        }
      case _ =>
    }
    */
  }

  /**
    * basic refcounting; when a function needs a buffer instead of a plain Real
    * (i.e. when it has potentially more than one use of the value), it further defers
    * the backpropagation of the gradient.  When all references have completed, can the
    * gradient be propagated further backwards.
    */
  override def buffer: VBuffer[D, S] = {
//    println(s"  BUFFERING  $this")
    refCount += 1
    this
  }

  override def complete(): Unit = {
//    println(s"  COMPLETING $this")
    refCount -= 1
    if (refCount == 0) {
      grad.foreach { g =>
        upstream.dv(g)
      }
      grad = None
    }
  }

  override def toString: String = {
    s"Buffer($upstream)"
  }
}

case class VNegate[D, S](upstream: Value[D, S])
    extends Value[D, S] {

  override def field: BaseField[D,S] =
    upstream.field

  override def shape: S = 
    upstream.shape

  override val v: D =
    field.negate(upstream.v)

  override def dv(dv: D): Unit =
      upstream.dv(field.negate(dv))

  override def toString: String = {
    s"- $upstream"
  }
}

case class VPlus[D, S](left: Value[D, S], right: Value[D, S])
    extends Value[D, S] {

  assert(left.shape == right.shape)

  override def field: BaseField[D, S] = left.field

  override def shape: S = 
    left.shape

  override val v: D = {
    val lt = left.v
    val rt = right.v
    field.plus(lt, rt)
  }

  override def dv(dv: D): Unit = {
    left.dv(dv)
    right.dv(dv)
  }

  override def toString: String = {
    s"($left + $right)"
  }
}

case class VMinus[D, S](left: Value[D, S], right: Value[D, S])
    extends Value[D, S] {

  assert(left.shape == right.shape)

  override def field: BaseField[D, S] =
    left.field

  override def shape: S = 
    left.shape

  override val v: D = {
    val lt = left.v
    val rt = right.v
    field.minus(lt, rt)
  }

  override def dv(dv: D): Unit = {
    left.dv(dv)
    right.dv(field.negate(dv))
  }

  override def toString: String = {
    s"($left - $right)"
  }
}

case class VTimes[D, S](left: Value[D, S], right: Value[D, S])
    extends Value[D, S] {

  assert(left.shape == right.shape)

  override def field: BaseField[D, S] =
    left.field

  override def shape: S = 
    left.shape

  override val v: D = {
    val lt = left.v
    val rt = right.v
    field.times(lt, rt)
  }

  override def dv(gradient:  D): Unit = {
    left.dv(field.times(gradient, right.v))
    right.dv(field.times(gradient, left.v))
  }

  override def toString: String = {
    s"($left * $right)"
  }
}

case class VDiv[D, S](left: Value[D, S], right: Value[D, S])
    extends Value[D, S] {

  assert(left.shape == right.shape)

  override def field: BaseField[D, S] =
    left.field

  override def shape: S = 
    left.shape

  override val v: D = {
    val nt = left.v
    val dt = right.v
    field.div(nt, dt)
  }

  override def dv(dv: D): Unit = {
    val dt = dv
    val dent = right.v

    left.dv(field.div(dv, dent))
    right.dv(field.div(
      field.times(dv, v),
      field.negate(dent)
    ))
  }

  override def toString: String = {
    s"($left / $right)"
  }
}