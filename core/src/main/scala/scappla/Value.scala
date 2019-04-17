package scappla

import scala.util.Random

trait Value[X] {

  def v: X

  def dv(v: X): Unit

  def unary_-()(implicit numE: Numeric[Value[X]]): Value[X] =
    numE.negate(this)

  def +(other: Value[X])(implicit numE: Numeric[Value[X]]): Value[X] =
    numE.plus(this, other)

  def -(other: Value[X])(implicit numE: Numeric[Value[X]]): Value[X] =
    numE.minus(this, other)

  def *(other: Value[X])(implicit numE: Numeric[Value[X]]): Value[X] =
    numE.times(this, other)

  def /(other: Value[X])(implicit numE: Fractional[Value[X]]): Value[X] =
    numE.div(this, other)

  def buffer: Buffered[X] =
    NoopBuffer(this)

  def const: Value[X] =
    Constant(this.v)
}

object Value {

  implicit def fromDouble(value: Double) = Real(value)

  implicit def mkNumericOps[X](value: Value[X])(implicit num: Fractional[Value[X]]) =
    num.mkNumericOps(value)
}

trait Buffered[X] extends Value[X] with Completeable

case class NoopBuffer[X](upstream: Value[X]) extends Buffered[X] {

  override def v: X = upstream.v

  override def dv(v: X): Unit = upstream.dv(v)

  override def buffer: Buffered[X] = this

  override def complete(): Unit = {}
}

class Constant[X](val v: X) extends Buffered[X] {

  override def dv(v: X): Unit = {}

  override def buffer: Buffered[X] = this

  override def complete(): Unit = {}
}

object Constant {

  def apply[X](v: X): Constant[X] = new Constant(v)
}

trait InferField[X, S] extends Fractional[Value[X]] {

  def const(x: X): Value[X]

  def fromInt(x: Int, shape: S): Value[X]

  def buffer(ex: Value[X]): Buffered[X]
}

object InferField {

  implicit val scalarNumeric: InferField[Double, Unit] =
    new InferField[Double, Unit] {

      // Ordering

      override def compare(x: Real, y: Real): Int = {
        x.v.compareTo(y.v)
      }

      // Numeric

      override def plus(x: Real, y: Real): BaseReal = DAdd(x, y)

      override def minus(x: Real, y: Real): BaseReal = DSub(x, y)

      override def times(x: Real, y: Real): BaseReal = DMul(x, y)

      override def negate(x: Real): BaseReal = DNeg(x)

      override def fromInt(x: Int): BaseReal = Real(x)

      override def toInt(x: Real): Int = x.v.toInt

      override def toLong(x: Real): Long = x.v.toLong

      override def toFloat(x: Real): Float = x.v.toFloat

      override def toDouble(x: Real): Double = x.v

      // Fractional

      override def div(x: Real, y: Real): BaseReal = DDiv(x, y)

      // InferField

      override def const(x: Double): Value[Double] = Real(x)

      override def fromInt(x: Int, shape: Unit): Value[Double] = Real(x)

      override def buffer(ex: Value[Double]) = RealBuffer(ex)
    }
}

trait BaseField[X, S] extends Fractional[X] {

  def shapeOf(x: X): S

  def fromInt(x: Int, shape: S): X

  def fromDouble(x: Double, shape: S): X

  def sqrt(x: X): X

  def gaussian(shape: S): X
}

object BaseField {

  implicit val doubleBaseField: BaseField[Double, Unit] =
    new BaseField[Double, Unit] {

      override def shapeOf(x: Double): Unit = Unit

      override def fromInt(x: Int, shape: Unit): Double = x.toDouble

      override def fromDouble(x: Double, shape: Unit): Double = x

      override def sqrt(x: Double): Double = math.sqrt(x)

      override def gaussian(shape: Unit): Double = Random.nextGaussian()

      override def div(x: Double, y: Double): Double = x / y

      override def plus(x: Double, y: Double): Double = x + y

      override def minus(x: Double, y: Double): Double = x - y

      override def times(x: Double, y: Double): Double = x * y

      override def negate(x: Double): Double = -x

      override def fromInt(x: Int): Double = x

      override def toInt(x: Double): Int = x.toInt

      override def toLong(x: Double): Long = x.toLong

      override def toFloat(x: Double): Float = x.toFloat

      override def toDouble(x: Double): Double = x

      override def compare(x: Double, y: Double): Int = java.lang.Double.compare(x, y)
    }

}
