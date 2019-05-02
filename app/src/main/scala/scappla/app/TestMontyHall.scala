package scappla.app

import scappla.Functions.{exp, sigmoid}
import scappla._
import scappla.distributions.{Bernoulli, Normal}
import scappla.guides.ReparamGuide
import scappla.optimization.Adam

import scala.util.Random

object TestMontyHall extends App {

  sealed trait Strategy
  case object Switch extends Strategy
  case object Remain extends Strategy

  var history = Seq.empty[(Strategy, Boolean)]

  class State {
    private var prior_pos: Double = 0.0
    private var prior_var: Double = 0.0
    private val posterior_pos = Param(0.0)
    private val posterior_var = Param(0.0)

    val guide = ReparamGuide(Normal(posterior_pos, exp(posterior_var)))

    def prior = Normal(prior_pos, exp(prior_var))

    def updatePrior(interpreter: Interpreter, lr: Double): Unit = {
      prior_pos += (interpreter.eval(posterior_pos).v - prior_pos) * lr
      prior_var += (interpreter.eval(posterior_var).v - prior_var) * lr
    }
  }

  val switch = new State
  val remain = new State

  val doors = 0 until 3
  val model: Model[((Real, Real), (Strategy, Boolean))] = infer {

    // sample probabilities of winning for both strategies
    val p_switch = sigmoid(sample(switch.prior, switch.guide))
    val p_remain = sigmoid(sample(remain.prior, remain.guide))

    /* Process history */

    for {(strategy, result) <- history} {
      strategy match {
        case Switch => observe(Bernoulli(p_switch), result)
        case Remain => observe(Bernoulli(p_remain), result)
      }
    }

    /* Create a new sample */

    // put price behind random door, assume contestant picks the first door
    val doorWithPrice = Random.nextInt(3)
    val selectedDoor = 0

    // let monty open one of the remaining doors
    val montyOptions = doors.filter { door =>
      door != doorWithPrice && door != selectedDoor
    }
    val montyOpens = montyOptions(Random.nextInt(montyOptions.size))

    if (p_switch.v > p_remain.v) {
      val switchedDoor = doors.filter { door =>
        door != selectedDoor && door != montyOpens
      }.head
      val haveWon = switchedDoor == doorWithPrice
      observe(Bernoulli(p_switch), haveWon)
      ((p_switch, p_remain), (Switch, haveWon))
    } else {
      val haveWon = selectedDoor == doorWithPrice
      observe(Bernoulli(p_remain), haveWon)
      ((p_switch, p_remain), (Remain, haveWon))
    }
  }

  val HISTORY_SIZE = 100
  val N = 1000

  val optimizer = new Adam(0.1)
  val interpreter = new OptimizingInterpreter(optimizer)

  // burn in
  for {_ <- 0 to N} {
    interpreter.reset()
    val ((p_switch, p_remain), (action, result)) = model.sample(interpreter)
    history = (action, result) +: history
    if (history.size > HISTORY_SIZE) {
      history = history.dropRight(1)
      switch.updatePrior(interpreter, 1.0 / HISTORY_SIZE)
      remain.updatePrior(interpreter, 1.0 / HISTORY_SIZE)
    }
    println(s"${p_switch.v}, ${p_remain.v}")
  }

  // measure
  /*
  val n_switch = Range(0, 1000).map { _ =>
    model.sample()
  }.count(_._1 == Switch)

  println(s"Nr switches: $n_switch")
  */
}
