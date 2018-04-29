package scappla

import scala.language.experimental.macros

trait DValue[X] {

  def v: X

  def dv(v: X): Unit

  def complete(): Unit = {}
}

trait DFunction1[From, To] extends ((From) => To) {

  def apply(in: DValue[From]): DValue[To]
}

trait DFunction2[From1, From2, To] extends ((From1, From2) => To) {

  def apply(in1: DValue[From1], in2: DValue[From2]): DValue[To]
}

class DScalar(self: DValue[Double]) {

  def unary_- = new DValue[Double] {

    override def v: Double = -self.v

    override def dv(v: Double): Unit = self.dv(-v)
  }

  def -(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v - other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad)
      other.dv(-grad)
    }

  }

  def +(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v + other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad)
      other.dv(grad)
    }
  }

  def *(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v * other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad * other.v)
      other.dv(grad * self.v)
    }
  }

  def /(other: DValue[Double]) = new DValue[Double] {

    private var grad = 0.0

    override lazy val v: Double = self.v / other.v

    override def dv(v: Double): Unit = {
      grad += v
    }

    override def complete(): Unit = {
      self.dv(grad / other.v)
      other.dv(-grad * v / other.v)
    }
  }

}

class DVariable(var v: Double) extends DValue[Double] {

  var grad = 0.0

  override def dv(v: Double): Unit = {
    grad += v
  }

}

class DConstant(val v: Double) extends DValue[Double] {

  override def dv(v: Double): Unit = {}
}

object Functions {

  object log extends DFunction1[Double, Double] {

    def apply(x: Double): Double = scala.math.log(x)

    // returned value takes ownership of the reference passed on the stack
    def apply(x: DValue[Double]): DValue[Double] = new DValue[Double] {

      private var grad = 0.0

      override lazy val v: Double = scala.math.log(x.v)

      override def dv(dx: Double): Unit = {
        grad += dx
      }

      override def complete(): Unit = {
        x.dv(grad / x.v)
      }
    }
  }

  object exp extends DFunction1[Double, Double] {

    override def apply(x: Double): Double = scala.math.exp(x)

    override def apply(x: DValue[Double]): DValue[Double] = new DValue[Double] {

      private var grad = 0.0

      override lazy val v: Double = scala.math.exp(x.v)

      override def dv(dx: Double): Unit = {
        grad += dx
      }

      override def complete(): Unit = {
        x.dv(grad * v)
      }
    }

  }

  object pow extends DFunction2[Double, Double, Double] {

    def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)

    def apply(base: DValue[Double], exp: DValue[Double]) = new DValue[Double] {

      private var grad = 0.0

      override lazy val v: Double = scala.math.pow(base.v, exp.v)

      override def dv(dx: Double): Unit = {
        grad += dx
      }

      override def complete(): Unit = {
        base.dv(grad * exp.v * scala.math.pow(base.v, exp.v - 1))
        exp.dv(grad * scala.math.log(base.v) * v)
      }

    }
  }

}

object DValue {

  implicit def toConstant(value: Double) = new DConstant(value)

  implicit def toScalarOps(value: DValue[Double]): DScalar = {
    new DScalar(value)
  }

  implicit val scalarOrdering: Ordering[DValue[Double]] = Ordering.by(_.v)

  implicit val scalarNumeric: Numeric[DValue[Double]] = new Numeric[DValue[Double]] {

    override def plus(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) + y

    override def minus(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) - y

    override def times(x: DValue[Double], y: DValue[Double]): DValue[Double] = toScalarOps(x) * y

    override def negate(x: DValue[Double]): DValue[Double] = -toScalarOps(x)

    override def fromInt(x: Int): DValue[Double] = toConstant(x)

    override def toInt(x: DValue[Double]): Int = x.v.toInt

    override def toLong(x: DValue[Double]): Long = x.v.toLong

    override def toFloat(x: DValue[Double]): Float = x.v.toFloat

    override def toDouble(x: DValue[Double]): Double = x.v

    override def compare(x: DValue[Double], y: DValue[Double]): Int = {
      x.v.compareTo(y.v)
    }
  }
}
