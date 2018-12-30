package scappla

import java.io.InputStreamReader

import com.github.tototoshi.csv.CSVReader
import scappla.Functions.exp
import scappla.distributions.Normal
import scappla.guides.ReparamGuide
import scappla.optimization.Adam

object TestChickweight extends App {

  import Real._

  val reader = CSVReader.open(new InputStreamReader(getClass.getResourceAsStream("/chickweight.csv")))
  val raw = reader.allWithHeaders()

  case class Record(weight: Double, time: Double, diet: Int)

  val data = raw.map { row =>
    Record(row("weight").toDouble, row("Time").toDouble, row("Diet").toInt)
  }


  val sgd = new Adam()
  val aPost = ReparamGuide(Normal(sgd.param(40.0), exp(sgd.param(0.0))))
  val atPost = ReparamGuide(Normal(sgd.param(0.0), exp(sgd.param(0.0))))

  val bPost = Range(1, 5).map { i =>
    i -> (
        ReparamGuide(Normal(sgd.param(10.0), exp(sgd.param(1.0)))),
        ReparamGuide(Normal(sgd.param(0.0), exp(sgd.param(0.0))))
    )
  }.toMap

  val model = infer {
    val a = sample(Normal(40.0, 1.0), aPost)
    val a_t = exp(sample(Normal(0.0, 0.2), atPost))
    val b = bPost.map { case (i, (muGuide, sGuide)) =>
      i -> (
          sample(Normal(10.0, 3.0), muGuide),
          exp(sample(Normal(0.0, 1.0), sGuide))
      )
    }

    data.foreach[Unit] {
      entry: Record =>
        val (b_mu, b_s) = b(entry.diet)
        observe(Normal(
          a + b_mu * entry.time, a_t + b_s * entry.time
        ), entry.weight: Real)
    }

    (a, a_t, b)
  }

  // warm up
  val N = 1000
  val startTime = System.currentTimeMillis()
  println("a,a_t,b1m,b1s,b2m,b2s,b3m,b3s,b4m,b4s")
  Range(0, N).foreach { i =>
    val (v_a, v_a_t, v_b) = sample(model)

    println(s"${v_a.v}, ${v_a_t.v}, ${
      v_b.toSeq.sortBy(_._1).map(_._2).map {
        case (m, s) => s"${m.v}, ${s.v}"
      }.mkString(", ")
    }")
  }
  val endTime = System.currentTimeMillis()
  println(s"Time: ${endTime - startTime} (avg: ${(endTime - startTime) * 1000.0 / N} mus / sample)")

}
