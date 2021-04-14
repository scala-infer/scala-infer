package scappla

object Functions {

  trait Op1 {

    trait Apply[-A, B] extends Function1[A, B]

    def apply[A, B](in: A)(implicit fn: Apply[A, B]): B = {
      fn.apply(in)
    }

    implicit def exprApply[A, AS, B, BS](implicit fn: Apply[Value[A, AS], Value[B, BS]]): Apply[Expr[A, AS], Expr[B, BS]] =
      new Apply[Expr[A, AS], Expr[B, BS]] {
        override def apply(in: Expr[A, AS]): Expr[B, BS] = Apply1(in, fn)
      }
  }

  trait Op2 {

    trait Apply[-A, -B, C] extends ((A, B) => C)

    def apply[A, B, C](a: A, b: B)(implicit fn: Apply[A, B, C]): C = {
      fn.apply(a, b)
    }

    implicit def exprApply[A, AS, B, BS, C, CS](implicit fn: Apply[Value[A, AS], Value[B, BS], Value[C, CS]]):
    Apply[Expr[A, AS], Expr[B, BS], Expr[C, CS]] = new Apply[Expr[A, AS], Expr[B, BS], Expr[C, CS]] {
      override def apply(a: Expr[A, AS], b: Expr[B, BS]): Expr[C, CS] = Apply2(a, b, fn)
    }
  }

  object log extends Op1 {

    implicit val logDouble: Apply[Double, Double] = new Apply[Double, Double] {
      override def apply(x: Double): Double = scala.math.log(x)
    }

    implicit def logValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      def apply(x: Value[D, S]): Value[D, S] = VLog(x)
    }

    case class VLog[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field: BaseField[D, S] =
        upstream.field

      override def shape: S =
        upstream.shape

      override val v: D = {
        field.log(upstream.v)
      }

      override def dv(dv: D): Unit = {
        upstream.dv(
          field.div(dv, upstream.v)
        )
      }

      override def toString: String = {
        s"log($upstream)"
      }
    }

  }

  object exp extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double) = scala.math.exp(x)
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      override def apply(x: Value[D, S]): Value[D, S] = TExp(x)
    }

    case class TExp[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.exp(upstream.v)
      }

      override def dv(dv: D): Unit = {
        val tv = this.v
        upstream.dv(field.times(dv, tv))
      }

      override def toString: String = {
        s"exp($upstream)"
      }
    }

  }

  object sqrt extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double) = scala.math.sqrt(x)
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      override def apply(x: Value[D, S]): Value[D, S] = TSqrt(x)
    }

    case class TSqrt[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.sqrt(upstream.v)
      }

      override def dv(dv: D): Unit = {
        val tv = this.v
        upstream.dv(field.div(
          dv,
          field.times(
            field.fromInt(2, shape),
            tv
          )
        ))
      }

      override def toString: String = {
        s"sqrt($upstream)"
      }
    }

  }


  object squared extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double) = x * x
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      override def apply(x: Value[D, S]): Value[D, S] = TSquared(x)
    }

    case class TSquared[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.times(upstream.v, upstream.v)
      }

      override def dv(dv: D): Unit = {
        upstream.dv(
          field.times(
            field.times(
              field.fromInt(2, shape),
              dv
            ),
            upstream.v
          )
        )
      }

      override def toString: String = {
        s"squared($upstream)"
      }
    }

  }


  object logistic extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      def apply(x: Value[D, S]): Value[D, S] = VLogistic(x)
    }

    case class VLogistic[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.logistic(upstream.v)
      }

      override def dv(dv: D): Unit = {
        val tv = this.v
        upstream.dv(field.times(
          dv,
          field.times(
            tv,
            field.logistic(field.negate(upstream.v))
          )
        ))
      }

      override def toString: String = {
        s"logistic($upstream)"
      }
    }

  }

  object tanh extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = math.tanh(x)
    }

    // (e^x - e^-x) / (e^x + e^-x) = 2 * logistic(2 * x) - 1
    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      def apply(x: Value[D, S]): Value[D, S] = {
        val field = x.field
        val two = Constant(field.fromInt(2, x.shape), x.shape)(field)
        val one = Constant(field.fromInt(1, x.shape), x.shape)(field)
        VMinus(
          VTimes(two, logistic(VTimes(two, x))),
          one
        )
      }
    }
  }

  object softplus extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = math.log1p(math.exp(x))
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      def apply(x: Value[D, S]): Value[D, S] = VSoftPlus(x)
    }

    case class VSoftPlus[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.softplus(upstream.v)
      }

      override def dv(dv: D): Unit = {
        val tv = this.v
        upstream.dv(field.times(
          dv,
          field.logistic(field.negate(upstream.v))
        ))
      }

      override def toString: String = {
        s"softplus($upstream)"
      }
    }


  }

  object lgamma extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(x: Double): Double = breeze.numerics.lgamma(x)
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S]] {
      def apply(x: Value[D, S]): Value[D, S] = VLGamma(x)
    }

    case class VLGamma[D, S](upstream: Value[D, S]) extends Value[D, S] {

      override def field = upstream.field

      override def shape = upstream.shape

      override val v: D = {
        field.lgamma(upstream.v)
      }

      override def dv(dv: D): Unit = {
        val tv = this.v
        upstream.dv(field.times(
          dv,
          field.digamma(v)
        ))
      }

      override def toString: String = {
        s"lgamma($upstream)"
      }
    }

  }

  object pow extends Op2 {

    implicit val forDouble: Apply[Double, Double, Double] = new Apply[Double, Double, Double] {
      def apply(base: Double, exp: Double): Double = scala.math.pow(base, exp)
    }

    implicit def forValue[D, S]: Apply[Value[D, S], Value[D, S], Value[D, S]] = new Apply[Value[D, S], Value[D, S], Value[D, S]] {
      def apply(base: Value[D, S], exp: Value[D, S]) = TPow(base, exp)
    }

    case class TPow[D, S](base: Value[D, S], expo: Value[D, S]) extends Value[D, S] {

      assert(base.shape == expo.shape)

      override def field =
        base.field

      override def shape =
        base.shape

      override val v: D = {
        val nt = base.v
        val dt = expo.v
        field.pow(nt, dt)
      }

      override def dv(dx: D): Unit = {
        base.dv(
          field.times(
            field.times(dx, expo.v),
            field.pow(
              base.v,
              field.minus(
                expo.v, field.fromInt(1, shape)
              )
            )
          )
        )
        expo.dv(
          field.times(
            field.times(dx, v),
            field.log(base.v)
          )
        )
      }

      override def toString: String = {
        s"($base ^ $expo)"
      }
    }

  }

  object sum extends Op1 {

    implicit val forDouble: Apply[Double, Double] = new Apply[Double, Double] {
      def apply(value: Double): Double = value
    }

    implicit val forReal: Apply[Real, Real] = new Apply[Real, Real] {
      override def apply(value: Real): Real = value
    }

  }

}
