import scala.util.Random

import scala.language.experimental.macros
import scappla.DValue._
import scappla.Functions._

package object scappla {

  type Score = DValue[Double]

  def autodiff[A, B](fn: A => B): DFunction1[A, B] =
    macro Macros.autodiff[A, B]

  //@formatter:off
  /**
    * The model is defined using the probability monad.  This follows the writer monad
    * pattern, such that an inference algorithm has access to a (noisy) estimate of the
    * posterior.
    *
    *   infer {
    *     val x = sample(variable)
    *     func(x)
    *   }
    *
    * gets rewritten to:
    *
    *   variable.map { x =>
    *     func(x)
    *   }
    *
    * With multiple samplings:
    *
    *   infer {
    *     val x = sample(varX)
    *     val y = sample(varY)
    *     func(x, y)
    *   }
    *
    * becomes:
    *
    *   varX.flatMap { x =>
    *     varY.map { y =>
    *       func(x, y)
    *     }
    *   }
    **/
  //@formatter:on

  /*
  def infer[X](fn: => X): Variable[X] =
    macro Macros.infer[X]

  def infer[Y, X](fn: Y => X): Y => Variable[X] =
    macro Macros.infer[X]
  */

  def sample[X](variable: Variable[X]): X =
    variable.sample()._1

  def factor(score: Score): Unit = {}

  // probability monad - collects scores for inference
  // Is used internally for bookkeeping purposes, should not be used in client code
  trait Variable[A] { self =>

    def sample(): (A, Score)

    def map[B](fn: A => B): Variable[B] = () => {
      val (a, score) = self.sample()
      (fn(a), score)
    }

    def flatMap[B](fn: A => Variable[B]): Variable[B] = () => {
        val (a, aScore) = self.sample()
        val (b, bScore) = fn(a).sample()
        (b, aScore + bScore)
      }
  }

  trait Distribution[A] {

    def draw(inference: Inference): Variable[A]

    def score(a: A): Score

    def support: Seq[A]
  }

  trait Inference {

    def infer[A](d: Distribution[A], v: Variable[A]): Variable[A]
  }

  case object Enumeration extends Inference {

    override def infer[A](dist: Distribution[A], prior: Variable[A]): Variable[A] =
      new Variable[A]() {

        override def sample(): (A, Score) =
          prior.sample()

        override def flatMap[B](fn: A => Variable[B]): Variable[B] = {
          val values = dist.support.map { v =>
            (dist.score(v), fn(v))
          }
          () => {
              val samples = values.map { case (scoreA, varB) =>
                val (b, scoreB) = varB.sample()
                (b, scoreA, scoreB)
              }

              // sample from softmax
              val probs = samples
                  .map { case (b, scoreA, _) =>
                    (b, exp(scoreA))
                  }
              val total = probs.map {
                _._2
              }.sum

              var draw = Random.nextDouble() * total.v
              var remaining = probs.toList
              while (remaining.head._2.v < draw) {
                draw = draw - remaining.head._2.v
                remaining = remaining.tail
              }

              val head = remaining.headOption.getOrElse(probs.head)
              (head._1, log(head._2 / total))
            }
        }
      }
  }

  case class Bernoulli(p: DValue[Double]) extends Distribution[Boolean] {
    self =>

    // sample from prior, i.e. based on p
    override def draw(inference: Inference): Variable[Boolean] =
      inference.infer(this, () => {
          val value = Random.nextDouble() < p.v
          (value, self.score(value))
        })

    override def score(value: Boolean): Score =
      new DValue[Double] {

        override lazy val v: Double =
          if (value) math.log(p.v) else math.log(1.0 - p.v)

        override def dv(d: Double): Unit =
          if (value) p.dv(d / p.v) else p.dv(-d / (1.0 - p.v))
      }

    override def support: Seq[Boolean] =
      Seq(true, false)
  }

}
