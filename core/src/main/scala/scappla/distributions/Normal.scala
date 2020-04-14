package scappla.distributions

import scappla.Functions.{log, squared, sum}
import scappla._

case class Normal[@specialized(Float, Double) D, S](
    mu: Expr[D, S],
    sigma: Expr[D, S]
)(implicit
    numX: BaseField[D, S],
    logImpl: log.Apply[Value[D, S], Value[D, S]],
    sumImpl: sum.Apply[Value[D, S], Value[Double, Unit]]
) extends DDistribution[D, S] {

  override def sample(interpreter: Interpreter): Buffered[D, S] = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)

    val shp = sigmaVal.shape
    val two = Constant(numX.fromInt(2, shp), shp)
    new Value[D, S] with Buffered[D, S] {
      var refCount = 1

      val e: Value[D, S] = Constant(numX.gaussian(shp), shp)

      val r: Buffered[D, S] = {
        val sc = sigmaVal.const
        ((muVal + e * sigmaVal / two) / (sc * sc)).buffer
      }

      override def field = numX

      override def shape = shp

      override val v: D =
        (muVal + e * sigmaVal).v

      override def dv(x: D): Unit = {
        if (refCount <= 0) {
          new Exception().printStackTrace()
          assert(refCount > 0)
        }
        r.dv(x)
      }

      override def buffer: Buffered[D, S] = {
        refCount += 1
        this
      }

      override def complete(): Unit = {
        assert(refCount > 0)
        refCount -= 1
        if (refCount == 0) {
          r.complete()
        }
      }

      private val id = hashCode()

      override def toString: String = {
        s"NormalSample(${id})"
      }
    }
  }

  override def observe(interpreter: Interpreter, x: Value[D, S]): Score = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)
    val shape = sigmaVal.shape
    val two = Constant(numX.fromInt(2, shape), shape)

    val e = (x - muVal) / sigmaVal
    sum(-log(sigmaVal) - squared(e) / two)
  }

  override def reparam_score(interpreter: Interpreter, x: Value[D, S]): Score = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)
    val shape = sigmaVal.shape
    val two = Constant(numX.fromInt(2, shape), shape)

    val e = (x - muVal.const) / sigmaVal.const
    sum(-log(sigmaVal).const - squared(e) / two)
  }
}
