import scala.util.Random
import scala.language.experimental.macros
import scappla.DValue._
import scappla.Functions._

import scala.collection.mutable

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
  trait Variable[A] {
    self =>

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

  def addFactor(score: Score): Variable[Unit] = {
    () => (Unit, score)
  }

  trait Distribution[A] {

    def draw(inference: Inference[A]): Variable[A]

    def score(a: A): Score

    def support: Seq[A]
  }

  trait Inference[A] {

    def infer(d: Distribution[A], v: Variable[A]): Variable[A]
  }

  case class NonInference[A]() extends Inference[A] {
    override def infer(d: Distribution[A], v: Variable[A]): Variable[A] = {
      v
    }
  }

  class Enumeration[A]() extends Inference[A] {

    override def infer(dist: Distribution[A], prior: Variable[A]): Variable[A] =
      new Variable[A]() {

        // running average of free energy for different choices of A
        // these approximations to the posterior inform the drawing process
        // NOTE: these actually depend on the history, i.e. ancestral draws
        // - this history-less approximation is the "mean-field"
        private val Zs: mutable.Map[A, (Int, Double)] =
          mutable.HashMap.empty.withDefaultValue((0, 0.0))

        override def sample(): (A, Score) = {
          val values = dist.support.map { v =>
            (v, dist.score(v))
          }
          sampleWithFree(values)
        }

        override def flatMap[B](fn: A => Variable[B]): Variable[B] = {
          val values = dist.support.map { v =>
            (v, fn(v))
          }
          () => {
            val samples = values.map { case (a, varB) =>
              val scoreA = dist.score(a)
              val (b, scoreB) = varB.sample()
              val (n, z) = Zs(a)
              if (n > 0)
                Zs(a) = (n + 1, z + math.log1p(math.exp(scoreB.v - z)) - math.log1p(1.0 / n))
              else
                Zs(a) = (1, scoreB.v)
              ((b, scoreB, z), scoreA + z)
            }
            val ((b, scoreB, z), totalZ) = sampleWithFree(samples)
            (b, totalZ - z + scoreB)
          }
        }

        // draw a sample, with the free energy as the score
        private def sampleWithFree[X](values: Seq[(X, DValue[Double])]) = {
          // sample from softmax
          val probs = values
              .map { case (b, scoreA) =>
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
          (head._1, total)
        }

      }
  }

  /*
  trait Optimizer[A] {

    def applyGradient(gradient: A): Unit
  }

  case class Variational[A: Numeric](guide: Variable[A], optimizer: Optimizer[A]) extends Inference[A] {

    override def infer(d: Distribution[A], prior: Variable[A]): Variable[A] =
      new Variable[A] {

        override def sample(): (A, Score) =
          prior.sample()

        override def flatMap[B](fn: A => Variable[B]): Variable[B] = {
          new Variable[B] {

            override def sample(): (B, Score) = {
              val (a, scoreGuide) = guide.sample()
              val scorePrior = d.score(a)
              val varB = fn(a)
              val (b, scoreB) = varB.sample()
              (b, onComplete(scorePrior + scoreB, _ => {
                scoreGuide.complete()
              }))
            }
          }
        }
      }
  }
  */

  case class Bernoulli(p: DValue[Double]) extends Distribution[Boolean] {
    self =>

    // sample from prior, i.e. based on p
    override def draw(inference: Inference[Boolean]): Variable[Boolean] =
      inference.infer(this, () => {
        val value = Random.nextDouble() < p.v
        (value, self.score(value))
      })

    override def score(value: Boolean): Score =
      if (value) log(p) else log(-p + 1.0)

    override def support: Seq[Boolean] =
      Seq(true, false)
  }

  /*
  case class Normal(mu: DValue[Double], sigma: DValue[Double]) extends Distribution[Double] {
    self =>

    override def draw(inference: Inference[Double]): Variable[Double] = {
      inference.infer(this, () => {
        val value = mu.v + sigma.v * Random.nextGaussian()
        (value, self.score(value))
      })
    }

    override def score(a: Double): Score = {
      val delta = (mu - a) / sigma
      -log(sigma) - delta * delta / 2.0
    }

    override def support: Seq[Double] = ???
  }
  */

}
