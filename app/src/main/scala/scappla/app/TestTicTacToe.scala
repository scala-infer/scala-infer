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
  val channel = Channel(20)

  type GridShape = Width :#: Height
  val gridShape = width :#: height

  type Position = Index[GridShape]
  type Grid = Map[Position, Occupant]

  type LayerShape = Width :#: Height :#: Channel
  val layerShape = width :#: height :#: channel

  val opt = new Adam(0.01)

  import Tensor._
  import TensorValue._

  private val gridField = implicitly[TensorField[GridShape, ArrayTensor]]
  private val layerField = implicitly[TensorField[LayerShape, ArrayTensor]]

  private def newTensor[S <: Shape](
    value: Float,
    shape: S
  )(implicit
    field: TensorField[S, ArrayTensor]
  ): Buffered[Tensor[S, ArrayTensor]] = {
    import field._
    opt.param(
      field.gaussian(shape) * field.fromDouble(value, shape)
    ).buffer
  }

  val toChannel = newTensor(0.1f, layerShape)
  val channelBias = newTensor(0.0f, channel)

  val toMus = newTensor(0.1f, layerShape)
  val muBias = newTensor(0f, gridShape)

  val toSigmas = newTensor(0.1f, layerShape)
  val sigmaBias = newTensor(-2f, gridShape)

  val guides = mutable.Map.empty[Grid, GridState]

  class GridState(val grid: Grid, val player: Occupant) {
    val (post_pos, post_var, hidden, mask, guide) = {
      val board = Array.ofDim[Float](9)
      val neither = Array.ofDim[Float](9)
      for {
        i <- 0 until width.size
        j <- 0 until height.size
      } {
        val index = Index[GridShape](List(i, j))
        grid(index) match {
          case Neither =>
            neither(3 * i + j) = 1f
          case `player` =>
            board(3 * i + j) = 1f
          case _ =>
            board(3 * i + j) = -1f
        }
      }
      val input = Tensor(
        gridShape,
        ArrayTensor(gridShape.sizes, board)
      )
      val mask = Tensor(
        gridShape,
        ArrayTensor(gridShape.sizes, neither)
      )
      val hidden = sigmoid(channelBias + (toChannel :*: input.const)).buffer
      val p_pos = (muBias + (hidden :*: toMus)).buffer
      val p_var = (sigmaBias + (hidden :*: toSigmas)).buffer
      (p_pos, p_var, hidden, mask, ReparamGuide(Normal(p_pos, exp(p_var))))
    }

    private var prior_pos = gridField.fromDouble(0.0, gridShape)
    private var prior_var = gridField.fromDouble(-2.0, gridShape)
    def prior: DDistribution[Tensor[GridShape, ArrayTensor]] =
      Normal[Tensor[GridShape, ArrayTensor], GridShape](
        prior_pos.const, exp(prior_var.const)
      )

    def complete(): Unit = {
      post_pos.complete()
      post_var.complete()
      hidden.complete()
      hidden.buffer
      post_pos.buffer
      post_var.buffer
    }

    def updatePrior(lr: Double): Unit = {
      val ops = implicitly[DataOps[ArrayTensor]]
      import gridField._
      prior_pos += (post_pos.v - prior_pos) * gridField.fromDouble(lr, gridShape)
      prior_var += (post_var.v - prior_var) * gridField.fromDouble(lr, gridShape)
    }

    def select(position: Position): GridState = {
      // println(mask.data.data.mkString(","))
      // println(s"Position: $position")
      assert(grid(position) == Neither)

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
        val scale = gridField.fromDouble(8.0, gridShape)
        val rates = sigmoid(scale * tanh(sample(state.prior, state.guide) / scale))
        // println("RATES: " + rates.v.data.data.mkString(","))
        val index = maxIndex(state.mask * rates)
        val rate = at(rates, index)
        val (result, sequence) = nextStep(state.select(index))
        observe(Bernoulli(rate), result == state.player)
        (result, state.grid +: sequence)
      }
    }
    nextStep(startState)
  }

  val seqs = Seq(
    Seq(toIdx(0, 0), toIdx(0, 1), toIdx(0, 2)),
    Seq(toIdx(1, 0), toIdx(1, 1), toIdx(1, 2)),
    Seq(toIdx(2, 0), toIdx(2, 1), toIdx(2, 2)),
    Seq(toIdx(0, 0), toIdx(1, 0), toIdx(2, 0)),
    Seq(toIdx(0, 0), toIdx(1, 0), toIdx(2, 0)),
    Seq(toIdx(0, 1), toIdx(1, 1), toIdx(2, 1)),
    Seq(toIdx(0, 2), toIdx(1, 2), toIdx(2, 2)),
    Seq(toIdx(0, 0), toIdx(1, 1), toIdx(2, 2)),
    Seq(toIdx(0, 2), toIdx(1, 1), toIdx(2, 0))
  )

  def toIdx(i: Int, j: Int): Position = {
    Index[GridShape](List(i,j))
  }

  def isFull(grid: Grid): Boolean =
    grid.size == gridShape.size

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

  for { _ <- Range(0, 100000)} {
    val (winner, sequence) = model.sample()
    muBias.complete()
    toMus.complete()
    sigmaBias.complete()
    toSigmas.complete()
    toChannel.complete()
    channelBias.complete()

    println(s"WINNER: ${winner} (${sequence.size})")
    // printGrid(sequence.last)

    for { grid <- sequence } {
      val state = guides(grid)
      state.updatePrior(0.1)
    }
    // println(s"MU BIAS: ${muBias.v.data}")

    // prepare for next sample
    muBias.buffer
    toMus.buffer
    sigmaBias.buffer
    toSigmas.buffer
    toChannel.buffer
    channelBias.buffer

  }
}