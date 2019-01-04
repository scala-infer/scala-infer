package scappla

import scappla.Functions.{exp, sigmoid}
import scappla.distributions.{Bernoulli, Normal}
import scappla.optimization.Adam

import scala.util.Random

object TestMontyHall extends App {

  import Real._

  sealed trait Action

  case object Switch extends Action

  case object Remain extends Action

  val optimizer = new Adam()
  val switch_dist = Normal(optimizer.param(0.0), exp(optimizer.param(0.0)))
  val remain_dist = Normal(optimizer.param(0.0), exp(optimizer.param(0.0)))

  val doors = 0 until 3

  val model: Model[Action] = infer {
    val doorWithPrice = Random.nextInt(3)
    val selectedDoor = 0

    val montyOptions = doors.filter { door =>
      door != doorWithPrice && door != selectedDoor
    }
    val montyOpens = montyOptions(Random.nextInt(montyOptions.size))

    val p_switch = sigmoid(switch_dist.sample())
    val p_remain = sigmoid(remain_dist.sample())
    if (p_switch.v > p_remain.v) {
      val switchedDoor = doors.filter { door =>
        door != selectedDoor && door != montyOpens
      }.head
      observe(Bernoulli(p_switch), switchedDoor == doorWithPrice)
      Switch
    } else {
      observe(Bernoulli(p_remain), selectedDoor == doorWithPrice)
      Remain
    }
  }

  val N = 10000
  // burn in
  for {_ <- 0 to N} {
    sample(model)
  }

  // measure
  val n_switch = Range(0, N).map { _ =>
    sample(model)
  }.count(_ == Switch)

  println(s"Nr switches: ${n_switch}")
}
