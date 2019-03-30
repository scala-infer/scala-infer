package scappla.app

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

  val gridShape = width :#: height

  type GridShape = Width :#: Height
  type Position = Index[GridShape]
  type Grid = Map[Position, Occupant]

  val opt = new Adam(0.1)

  import TensorExpr._

  val toChannelInit = TensorExpr.numTensor[Width :#: Height :#: Channel, ArrayTensor]
      .gaussian(width :#: height :#: channel)
  val toChannel = opt.param(toChannelInit)

  val toGridInit = TensorExpr.numTensor[Width :#: Height :#: Channel, ArrayTensor]
      .gaussian(width :#: height :#: channel)
  val toMus = opt.param(toGridInit)
  val toSigmas = opt.param(toGridInit)

  import TensorExpr._

  val guides = mutable.Map.empty[Grid, GridState]

  class GridState(
      val grid: Grid,
      val player: Occupant,
      init_pos: Tensor[GridShape, ArrayTensor] =
        Tensor(gridShape, ArrayTensor(List(3, 3), Array.fill(9)(0f))),
      init_var: Tensor[GridShape, ArrayTensor] =
        Tensor(gridShape, ArrayTensor(List(3, 3), Array.fill(9)(0f)))
  ) {
    val (post_pos, post_var, guide) = {
      val values = Array.ofDim[Float](9)
      for {
        i <- 0 until width.size
        j <- 0 until height.size
      } {
        val index = Index[Width :#: Height](List(i, j))
        values(i * 3 + j) = grid(index) match {
          case Neither => 0f
          case Cross => 1f
          case Circle => -1f
        }
      }
      val input = Tensor(
        gridShape,
        ArrayTensor(gridShape.sizes, values)
      )
      val hidden = sigmoid(toChannel :*: input.const)
      val p_pos = hidden :*: toMus
      val p_var = hidden :*: toSigmas
      (p_pos, p_var, ReparamGuide(Normal(p_pos, exp(p_var))))
    }

    private var prior_pos = init_pos
    private var prior_var = init_var
    def prior: DDistribution[Tensor[GridShape, ArrayTensor]] =
      Normal[Tensor[GridShape, ArrayTensor], GridShape](
        prior_pos.const, exp(prior_var.const)
      )

    def updatePrior(lr: Double): Unit = {
      val ops = implicitly[DataOps[ArrayTensor]]
      prior_pos = Tensor(
        gridShape,
        ops.times(
          ops.minus(
            post_pos.v.data,
            prior_pos.data
          ),
          ops.fill(lr.toFloat, gridShape.sizes: _*)
        )
      )
      prior_var = Tensor(
        gridShape,
        ops.times(
          ops.minus(
            post_var.v.data,
            prior_var.data
          ),
          ops.fill(lr.toFloat, gridShape.sizes: _*)
        )
      )
    }

    def select(position: Position): GridState = {
      val newGrid = grid.updated(position, player)
      if (!guides.contains(newGrid)) {
        val newState = new GridState(newGrid, player.flip)
        guides.update(newGrid, newState)
        newState
      } else {
        guides(newGrid)
      }
    }
  }

  val emptyGrid: Grid = Map.empty[Position, Occupant]
      .withDefaultValue(Neither)
  val startState = new GridState(emptyGrid, Cross)

  val model = infer {
    def nextStep(state: GridState): Occupant = {
      val w = winner(state.grid)
      if (w != Neither) {
        w
      } else if (isFull(state.grid)) {
        Neither
      } else {
        val rates = sample(state.prior, state.guide)
        val index = maxIndex(rates)
        nextStep(state.select(index))
      }
    }
    nextStep(startState)
  }

  def toIdx(i: Int, j: Int): Position = {
    Index[GridShape](List(i,j))
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

  def isFull(grid: Grid): Boolean =
    !grid.exists(_._2 == Neither)

  def winner(grid: Grid): Occupant = {
    seqs.foldLeft[Occupant](Neither) { case (cur, s) =>
      cur match {
        case Neither =>
          if (s.forall(grid(_) == Cross))
            Cross
          else if (s.forall(grid(_) == Circle))
            Circle
          else
            Neither
        case _ => cur
      }
    }
  }
}