package scappla

object Functions {

  trait Op1 {

    trait Apply[-A, B] extends Function1[A, B]

    def apply[A, B](in: A)(implicit fn: Apply[A, B]): B = {
      fn.apply(in)
    }
  }

  trait Op2 {

    trait Apply[-A, -B, C] extends ((A, B) => C)

    def apply[A, B, C](a: A, b: B)(implicit fn: Apply[A, B, C]): C = {
      fn.apply(a, b)
    }
  }

  object log extends Op1 {

    implicit val logDouble: Apply[Double, Double] = new Apply[Double, Double] {
      override def apply(x: Double): Double = scala.math.log(x)
    }

    implicit val logReal: Apply[Real, Real] = new Apply[Real, Real] {

      def apply(x: Real): Real = new LazyReal(0.0) {

        override def _v: Double =
          scala.math.log(x.v)

        override def _dv(dx: Double): Unit = {
          x.dv(dx / x.v)
        }

        override def toString: String = s"Log($x)"
      }
    }
  }

  object exp extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double) = scala.math.exp(x)
    }

    implicit val forReal: Apply[Real, Real] = new Apply[Real, Real] {
      override def apply(x: Real): Real = new LazyReal(0.0) {

        override def _v: Double =
          scala.math.exp(x.v)

        override def _dv(dx: Double): Unit = {
          x.dv(dx * v)
        }

        override def toString: String = s"Exp($x)"
      }
    }

  }

  object sigmoid extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    }

    implicit val forReal: Apply[Real, Real] = new Apply[Real, Real] {
      def apply(x: Real): Real =
        DDiv(Real(1.0), DAdd(exp(DNeg(x)), Real(1.0)))
    }
  }

  object tanh extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = math.tanh(x)
    }

    implicit val forReal: Apply[Real, Real] = new Apply[Real, Real] {
      def apply(x: Real): Real = {
        // (e^x - e^-x) / (e^x + e^-x)
        // = (1.0 - e^(-2x)) / (1 + e^(-2x))
        // = 2.0 / (1 + e^(-2x)) - 1
        DAdd(Real(-1.0), DDiv(Real(2.0), DAdd(exp(DNeg(DMul(Real(2.0), x))), Real(1.0))))
      }
    }
  }

  object pow extends Op2 {

    implicit val forDouble: Apply[Double, Double, Double] = new Apply[Double, Double, Double] {
      def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)
    }

    implicit val forReal: Apply[Real, Real, Real] = new Apply[Real, Real, Real] {
      def apply(base: Real, exp: Real) = new LazyReal(0.0) {

        override def _v: Double =
          scala.math.pow(base.v, exp.v)

        override def _dv(dx: Double): Unit = {
          val ev = exp.v
          base.dv(dx * ev * scala.math.pow(base.v, ev - 1))
          exp.dv(dx * scala.math.log(base.v) * v)
        }

        override def toString: String = s"Pow($base, $exp)"
      }
    }
  }

  object sum {

    trait Apply[-A] extends Function1[A, Real]

    def apply[A](in: A)(implicit fn: Apply[A]): Real = {
      fn.apply(in)
    }

    implicit val forDouble: Apply[Double] = new Apply[Double] {
      def apply(value: Double): Real = Real(value)
    }

    implicit val forReal: Apply[Real] = new Apply[Real] {
      override def apply(value: Score): Real = value
    }

  }

}
