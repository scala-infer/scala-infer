package scappla.app

import scappla.Functions.{exp, sigmoid}
import scappla.distributions.{Bernoulli, Normal}
import scappla.optimization.Adam
import scappla._
import scappla.guides.ReparamGuide

import scala.util.Random

object TestMontyHall extends App {

  import Real._

  sealed trait Action

  case object Switch extends Action

  case object Remain extends Action

  val optimizer = new Adam(0.1)

  var (switch_mu_prior, switch_s_prior) = (0.0, 0.0)
  val (switch_mu, switch_s) = (optimizer.param(0.0), optimizer.param(0.0))
  val switch_dist = ReparamGuide(Normal(switch_mu, exp(switch_s)))

  var (remain_mu_prior, remain_s_prior) = (0.0, 0.0)
  val (remain_mu, remain_s) = (optimizer.param(0.0), optimizer.param(0.0))
  val remain_dist = ReparamGuide(Normal(remain_mu, exp(remain_s)))

  var history = Seq.empty[(Action, Boolean)]

  val doors = 0 until 3

  val model: Model[(Action, Boolean)] = infer {
    val doorWithPrice = Random.nextInt(3)
    val selectedDoor = 0

    val montyOptions = doors.filter { door =>
      door != doorWithPrice && door != selectedDoor
    }
    val montyOpens = montyOptions(Random.nextInt(montyOptions.size))

    val p_switch = sigmoid(sample(Normal(switch_mu_prior, exp(switch_s_prior)), switch_dist))
    val p_remain = sigmoid(sample(Normal(remain_mu_prior, exp(remain_s_prior)), remain_dist))
    for {
      (action, result) <- history
    } {
      action match {
        case Switch =>
          observe(Bernoulli(p_switch), result)
        case Remain =>
          observe(Bernoulli(p_remain), result)
      }
    }
    if (p_switch.v > p_remain.v) {
      val switchedDoor = doors.filter { door =>
        door != selectedDoor && door != montyOpens
      }.head
      val result = switchedDoor == doorWithPrice
      observe(Bernoulli(p_switch), result)
      (Switch, result)
    } else {
      val result = selectedDoor == doorWithPrice
      observe(Bernoulli(p_remain), result)
      (Remain, result)
    }
  }

  val HISTORY_SIZE = 100

  val N = 1000
  // burn in
  for {_ <- 0 to N} {
    val (action, result) = model.sample()
    history = (action, result) +: history
    if (history.size > HISTORY_SIZE) {
      history = history.dropRight(1)
      switch_mu_prior += (switch_mu.v - switch_mu_prior) / HISTORY_SIZE
      switch_s_prior += (switch_s.v - switch_s_prior) / HISTORY_SIZE
      remain_mu_prior += (remain_mu.v - remain_mu_prior) / HISTORY_SIZE
      remain_s_prior += (remain_s.v - remain_s_prior) / HISTORY_SIZE
    }
    println(s"${sigmoid(switch_mu.v + exp(switch_s.v))}, ${sigmoid(switch_mu.v - exp(switch_s.v))} ${sigmoid(remain_mu.v + exp(remain_s.v))}, ${sigmoid(remain_mu.v - exp(remain_s.v))}")
  }

  // measure
  /*
  val n_switch = Range(0, 1000).map { _ =>
    model.sample()
  }.count(_._1 == Switch)

  println(s"Nr switches: $n_switch")
  */
}
