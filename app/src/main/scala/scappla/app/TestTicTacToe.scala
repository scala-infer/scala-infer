package scappla.app

import scala.util.Random

import scappla._
import scappla.Functions._
import scappla.guides._
import scappla.distributions._
import scappla.optimization._
import scappla.tensor._

import scala.collection.mutable

object TestTicTacToe extends App {

  sealed trait Occupant {

    def flip: Occupant
  }

  case object Neither extends Occupant {
    val flip = Neither
  }

  case object Cross extends Occupant {
    val flip = Circle
  }

  case object Circle extends Occupant {
    val flip = Cross
  }

  case class Width(size: Int) extends Dim[Width]
  val width = Width(3)

  case class Height(size: Int) extends Dim[Height]
  val height = Height(3)

  case class Channel(size: Int) extends Dim[Channel]
  val channel = Channel(50)

  type GridShape = Width :#: Height
  val gridShape = width :#: height

  type Position = Index[GridShape]
  type Grid = Map[Position, Occupant]

  type LayerShape = Width :#: Height :#: Channel
  val layerShape = width :#: height :#: channel

  import Tensor._

  private val gridField = implicitly[TensorField[ArrayTensor, GridShape]]
  private val layerField = implicitly[TensorField[ArrayTensor, LayerShape]]

  private def newTensor[S <: Shape](
    value: Float,
    shape: S
  )(implicit
    base: TensorField[ArrayTensor, S] //,
    // expr: ValueField[Tensor[S, ArrayTensor], S]
  ): Expr[ArrayTensor, S] = {
    import base._
    val init = base.times(base.gaussian(shape), base.fromDouble(value, shape))
    Param(init, shape)
  }

  val toChannel = newTensor(0.1f, layerShape)
  val channelBias = newTensor(0.0f, channel)

  val toMus = newTensor(0.1f, layerShape)
  val muBias = newTensor(0f, gridShape)

  val toSigmas = newTensor(0.1f, layerShape)
  val sigmaBias = newTensor(-4f, gridShape)

  val inputBias = Apply1(
    Param(0.0),
    (ib: Real) => broadcast(ib, gridShape)
  )

  val guides = mutable.Map.empty[Grid, GridState]

  class GridState(val grid: Grid, val player: Occupant) {
    val (post_pos, post_var, hidden, mask, guide) = {
      val board = Array.ofDim[Float](9)
      var neither: List[Index[GridShape]] = Nil
      for {
        i <- 0 until width.size
        j <- 0 until height.size
      } {
        val index = Index[GridShape](List(i, j))
        grid(index) match {
          case Neither =>
            neither = index :: neither
          case `player` =>
            board(3 * i + j) = 1f
          case _ =>
            board(3 * i + j) = -1f
        }
      }
      val input: Expr[ArrayTensor, GridShape] = Value(
        ArrayTensor(gridShape.sizes, board),
        gridShape
      )
      val hidden = tanh(channelBias + (toChannel :*: input))
      val p_pos = muBias + (hidden :*: toMus) + inputBias * input
      val p_var = sigmaBias + (hidden :*: toSigmas)
      (p_pos, p_var, hidden, neither, ReparamGuide(Normal(p_pos, softplus(p_var))))
    }

    def prior: DDistribution[ArrayTensor, GridShape] = {
      val prior_pos = Apply1(post_pos, (v: Value[ArrayTensor, GridShape]) => v.const)
      val prior_var = Apply1(post_var, (v: Value[ArrayTensor, GridShape]) => v.const)
      Normal[ArrayTensor, GridShape](prior_pos, softplus(prior_var))
    }

    /*
    def updatePrior(interpreter: Interpreter, lr: Double): Unit = {
      val ops = implicitly[TensorData[ArrayTensor]]
      import gridField._
      val pp_v = interpreter.eval(post_pos)
      prior_pos = gridField.plus(
        prior_pos,
        gridField.times(
          gridField.minus(pp_v.v, prior_pos),
          gridField.fromDouble(lr, gridShape)
        )
      )
      val pv_v = interpreter.eval(post_var)
      prior_var = gridField.plus(
        prior_var,
        gridField.times(
          gridField.minus(pv_v.v, prior_var),
          gridField.fromDouble(lr, gridShape)
        )
      )
    }
   */

    def maxIndex(samples: Value[ArrayTensor, GridShape]): Position = {
      mask.foldLeft[Option[(Double, Position)]](None) {
        case (current, index) =>
          val value = at(samples, index)
          current match {
            case None =>
              Some(value.v, index)
            case Some((curMax, _)) if value.v > curMax =>
              Some(value.v, index)
            case _ => current
          }
      } match {
        case Some((_, index)) => index
        case _ => throw new RuntimeException
      }
    }

    def select(position: Position): GridState = {
      // println(mask.data.data.mkString(","))
      // println(s"Position: $position")
      if (grid(position) != Neither) {
        printGrid(grid)
        println(s"POS: ${position}")
        assert(grid(position) == Neither)
      }

      val newGrid = grid + (position -> player)
      if (!guides.contains(newGrid)) {
        val newState = new GridState(newGrid, player.flip)
        guides += newGrid -> newState
        newState
      } else {
        guides(newGrid)
      }
    }
  }

  val emptyGrid: Grid = Map.empty[Position, Occupant]
      .withDefaultValue(Neither)
  val startState = new GridState(emptyGrid, Cross)
  guides += emptyGrid -> startState

  val model = infer {

    def nextStep(state: GridState): (Occupant, Seq[Grid]) = {
      val w = winner(state.grid)
      if (w != Neither) {
        // printGrid(state.grid)
        (w, Seq(state.grid))
      } else if (isFull(state.grid)) {
        // printGrid(state.grid)
        (Neither, Seq(state.grid))
      } else {
        val samples = sample(state.prior, state.guide)
//        println("SAMPLES: " + samples.v.data.mkString(","))
        val index = state.maxIndex(samples)
        val (result, sequence) = nextStep(state.select(index))
        if (result != Neither) {
          val rate = logistic(at(samples, index))
          observe(Bernoulli(rate), result == state.player)
        }
        (result, state.grid +: sequence)
      }
    }
    nextStep(startState)
  }

  def isFull(grid: Grid): Boolean =
    grid.size == gridShape.size

  val seqs = Seq(
    // vertical
    line((0, 0), (0, 1)),
    line((1, 0), (0, 1)),
    line((2, 0), (0, 1)),

    // horizontal
    line((0, 0), (1, 0)),
    line((0, 1), (1, 0)),
    line((0, 2), (1, 0)),

    // diagonal
    line((0, 0), (1, 1)),
    line((0, 2), (1, -1))
  )

  def line(from: (Int, Int), dir: (Int, Int)) = {
    for { dist <- 0 until 3 } yield
      Index[GridShape](List(
        from._1 + dir._1 * dist,
        from._2 + dir._2 * dist
      ))
  }

  def winner(grid: Grid): Occupant = {
    seqs.foldLeft[Occupant](Neither) { case (cur, s) =>
      cur match {
        case Neither =>
          if (s.forall(grid.get(_).contains(Cross)))
            Cross
          else if (s.forall(grid.get(_).contains(Circle)))
            Circle
          else
            Neither
        case _ => cur
      }
    }
  }

  def printGrid(grid: Grid): Unit = {
    for {
      i <- 0 until width.size
      j <- 0 until height.size
    } {
      val sym = grid(Index[GridShape](List(i, j))) match {
        case Neither => " "
        case Circle => "O"
        case Cross => "X"
      }
      print(s" $sym ")
      if (((j + 1) % width.size) == 0) {
        println()
      }
    }
  }

  def gridToString(grid: Grid): Seq[String] = {
    for {
      i <- 0 until width.size
    } yield {
      " " + (for {
        j <- 0 until height.size
      } yield {
        grid(Index[GridShape](List(i, j))) match {
          case Neither => " "
          case Circle => "O"
          case Cross => "X"
        }
      }).mkString(" ") + " "
    }
  }

  def printGridSequence(seq: Seq[Grid]): Unit = {
    val seqStrs = seq.map(gridToString)
    for {
      i <- 0 until width.size
    } {
      println(seqStrs.map(rows => rows(i)).mkString("|"))
    }
  }

  val opt = new Adam(0.005)
  val interpreter = new OptimizingInterpreter(opt)

  for { _ <- Range(0, 100000)} {
    interpreter.reset()
    val (winner, sequence) = model.sample(interpreter)

    println(s"WINNER: ${winner} (${sequence.size})")
    printGridSequence(sequence)

    /*
    for { grid <- sequence } {
      val state = guides(grid)
      state.updatePrior(interpreter, 0.1)
    }
     */
    // println(s"MU BIAS: ${muBias.v.data}")
  }
}