package scappla

// element-wise operations
trait Elemwise[D] {

  def plus(a: D, b: D): D

  def minus(a: D, b: D): D

  def times(a: D, b: D): D

  def div(a: D, b: D): D

  def pow(a: D, b: D): D

  def negate(a: D): D

  def sqrt(a: D): D

  def log(a: D): D

  def exp(a: D): D

  def logistic(a: D): D

  def softplus(a: D): D

  def sumAll(a: D): Float
}

object ElemwiseOps {

  implicit val doubleOps = new Elemwise[Double] {

    override def plus(a: Double, b: Double): Double =
      a + b

    override def minus(a: Double, b: Double): Double =
      a - b

    override def times(a: Double, b: Double): Double =
      a * b

    override def div(a: Double, b: Double): Double =
      a / b

    override def pow(a: Double, b: Double): Double =
      scala.math.pow(a, b)

    override def negate(a: Double): Double =
      -a

    override def sqrt(a: Double): Double =
      scala.math.sqrt(a)

    override def log(a: Double): Double =
      scala.math.log(a)

    override def exp(a: Double): Double =
      scala.math.exp(a)

    override def logistic(a: Double): Double =
      if (a > 0.0) {
        1.0 / (1.0 + scala.math.exp(-a))
      } else {
        val ea = scala.math.exp(a)
        ea / (1.0 + ea)
      }

    override def softplus(a: Double): Double =
      math.log1p(math.exp(a))

    override def sumAll(a: Double): Float =
      a.toFloat
  }

}
