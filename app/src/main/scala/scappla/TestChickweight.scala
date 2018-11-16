package scappla

import java.io.InputStreamReader

import com.github.tototoshi.csv.CSVReader
import scappla.Functions.{exp, log}
import scappla.distributions.Normal
import scappla.guides.ReparamGuide
import scappla.optimization.SGDMomentum

object TestChickweight extends App {

  import Real._

  val reader = CSVReader.open(new InputStreamReader(getClass.getResourceAsStream("/chickweight.csv")))
  val raw = reader.allWithHeaders()

  case class Record(weight: Double, time: Int, chick: Int, diet: Int)

  val data = raw.map { row =>
    Record(row("weight").toDouble, row("Time").toInt, row("Chick").toInt, row("Diet").toInt)
  }

  val sgd = new SGDMomentum(mass = 200)
  val aPost = ReparamGuide(Normal(sgd.param(40.0, 1.0), exp(sgd.param(0.0, 1.0))))
  val atPost = ReparamGuide(Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0))))

  val bPost = Range(1, 5).map { i =>
    i -> (
        ReparamGuide(Normal(sgd.param(7.0, 1.0), exp(sgd.param(0.0, 1.0)))),
        ReparamGuide(Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0))))
    )
  }.toMap

  val model = infer {
    val a = sample(Normal(40.0, 1.0), aPost)
    val a_t = exp(sample(Normal(0.0, 0.2), atPost))
    val b = bPost.map { case (i, (muGuide, sGuide)) =>
      i -> (
          sample(Normal(7.0, 2.0), muGuide),
          exp(sample(Normal(0.0, 1.0), sGuide))
      )
    }

    data.foreach[Unit] {
      entry: Record =>
        val (b_mu, b_s) = b(entry.diet)
        observe(Normal(
          a + b_mu * (entry.time * entry.time).toDouble / 20.0, a_t + b_s * entry.time.toDouble
        ), entry.weight: Real)
    }

    (a, a_t, b)
  }

  // warm up
  val N = 10000
  val startTime = System.currentTimeMillis()
  println("a,a_t,b1m,b1s,b2m,b2s,b3m,b3s,b4m,b4s")
  Range(0, N).foreach { i =>
    val (v_a, v_a_t, v_b) = sample(model)

    println(s"${v_a.v}, ${v_a_t.v}, ${
      v_b.toSeq.sortBy(_._1).map(_._2).map {
        case (m, s) => s"${m.v}, ${s.v}"
      }.mkString(",")
    }")
  }
  val endTime = System.currentTimeMillis()
  println(s"Time: ${endTime - startTime} (avg: ${(endTime - startTime) * 1000.0 / N} mus / sample)")

}
