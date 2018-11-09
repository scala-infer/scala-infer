package scappla.distributions

import scappla.Functions.log
import scappla.{Real, Score, DAdd}

import scala.util.Random

import Real._

case class Categorical(p: Seq[Real]) extends Distribution[Int] {

  private val total = p.reduce(DAdd)

  override def sample(): Sample[Int] = {
    val draw = Random.nextDouble() * total.v
    val (_, index) = p.zipWithIndex.foldLeft((draw, 0)) {
      case ((curDraw, curIdx), (p_i, idx)) =>
        val newDraw = curDraw - p_i.v
	if (curDraw > 0) {
	  (newDraw, idx)
	} else {
	  (newDraw, curIdx)
	}
    }
    new Sample[Int] {

      override val get: Int =
        index

      override val score: Score =
        Categorical.this.observe(get)

      override def complete(): Unit = {}
    }
  }

  override def observe(value: Int): Score = {
    log(p(value) / total)
  }

}

