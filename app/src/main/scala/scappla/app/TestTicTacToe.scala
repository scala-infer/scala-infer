package scappla.app

import scappla._
import scappla.Functions._
import scappla.guides._
import scappla.distributions._
import scappla.optimization._
import scappla.tensor._

object TestTicTacToe extends App {

  sealed trait Occupant

  case object Neither extends Occupant

  case object Cross extends Occupant

  case object Circle extends Occupant

  case class Width(size: Int) extends Dim[Width]
  val width = Width(3)

  case class Height(size: Int) extends Dim[Height]
  val height = Height(3)

  case class Channel(size: Int) extends Dim[Channel]
  val channel = Channel(20)

  val grid = (0 to 9).map { _ => Neither }
  val p = (0 to 9).map { _ => Real(1.0) }

  val opt = new Adam(0.1)
  val q = (0 to 9).map { _ => sigmoid(opt.param(1.0)) }
  val initialGuide = BBVIGuide(Categorical(q))

  import TensorExpr._

  val toChannelInit = TensorExpr.numTensor[Width :#: Height :#: Channel, ArrayTensor]
      .gaussian(width :#: height :#: channel)
  val toChannel = opt.param(toChannelInit)

  val toGridInit = TensorExpr.numTensor[Width :#: Height :#: Channel, ArrayTensor]
      .gaussian(width :#: height :#: channel)
  val toGrid = opt.param(toGridInit)

  val guides = Map.empty[Seq[Occupant], BBVIGuide[Int]]
      .withDefault(grid => {
        val values = grid.toArray.map {
          case Neither => 0f
          case Cross => 1f
          case Circle => -1f
        }
        val input = Tensor(
          width :#: height,
          ArrayTensor(Seq(width.size, height.size), values)
        )
        val hidden = sigmoid(toChannel :*: input.const)
        val p_grid = hidden :*: toGrid
        BBVIGuide(Categorical())
      })

  val model = infer {

    val nextStep = { grid: Seq[Occupant] =>
      val w = winner(grid)
      if (w != Neither) {
        w
      } else if (isFull(grid)) {
        Neither
      } else {
        val place = sample(Categorical(p), initialGuide)
        val newGrid = grid.updated(place, Cross)
        nextStep(newGrid)
      }
    }
  }

  val seqs = Seq(
    Seq(0, 1, 2),
    Seq(3, 4, 5),
    Seq(6, 7, 8),
    Seq(0, 3, 6),
    Seq(1, 4, 7),
    Seq(2, 5, 8),
    Seq(0, 4, 8),
    Seq(6, 4, 2)
  )

  def isFull(grid: Seq[Occupant]): Boolean = !grid.contains(Neither)

  def winner(grid: Seq[Occupant]): Occupant = {
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