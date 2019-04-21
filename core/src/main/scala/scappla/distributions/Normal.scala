package scappla.distributions

import scappla.Functions.{log, sum}
import scappla._

case class Normal[D, S](
    mu: Expr[D, S],
    sigma: Expr[D, S]
)(implicit
    numE: ValueField[D, S],
    numX: BaseField[D, S],
    logImpl: log.Apply[Value[D], Value[D]],
    sumImpl: sum.Apply[Value[D], Value[Double]]
) extends DDistribution[D] {

  override def sample(interpreter: Interpreter): Buffered[D] = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)

    val shape = numX.shapeOf(sigmaVal.v)
    val two = numE.fromInt(2, shape)
    new Value[D] with Buffered[D] {
      var refCount = 1

      val e: Value[D] = Constant(numX.gaussian(shape))

      val r: Buffered[D] = {
        val sc = sigmaVal.const
        ((muVal + e * sigmaVal / two) / (sc * sc)).buffer
      }

      override val v: D =
        (muVal + e * sigmaVal).v

      override def dv(x: D): Unit = {
        assert(refCount > 0)
        r.dv(x)
      }

      override def buffer: Buffered[D] = {
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

      override def toString: String = {
        s"NormalSample(${hashCode()})"
      }
    }
  }

  override def observe(interpreter: Interpreter, x: Value[D]): Score = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)
    val shape = numX.shapeOf(sigmaVal.v)
    val two = numE.fromInt(2, shape)

    val e = (x - muVal) / sigmaVal
    sum(-log(sigmaVal) - e * e / two)
  }

  override def reparam_score(interpreter: Interpreter, x: Value[D]): Score = {
    val sigmaVal = interpreter.eval(sigma)
    val muVal = interpreter.eval(mu)
    val shape = numX.shapeOf(sigmaVal.v)
    val two = numE.fromInt(2, shape)

    val e = (x - muVal.const) / sigmaVal.const
    sum(-log(sigmaVal).const - e * e / two)
  }
}
