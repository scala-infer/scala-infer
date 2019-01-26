package scappla.distributions

import scappla.Functions.log
import scappla.{Real, Score, DAdd}

import scala.util.Random

import scappla.Real._

case class Categorical(p: Seq[Real]) extends Distribution[Int] {

  import scappla.InferField._

  private val total = p.reduce(DAdd)

  override def sample(): Int = {
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
    index
  }

  override def observe(value: Int): Score = {
    log(p(value) / total)
  }

}

