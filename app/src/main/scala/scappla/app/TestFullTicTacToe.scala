package scappla.app

import scappla._
import scappla.guides._
import scappla.optimization._
import scappla.tensor._
import scappla.tensor.Tensor._
import scappla.distributions._
import scappla.Functions._

import scala.util.Random

object TestFullTicTacToe extends App {

  sealed trait Tag {
    def switch: Tag
  }
  case object Neither extends Tag {
    lazy val switch = Neither
  }
  case object Cross extends Tag {
    lazy val switch = Circle
  }
  case object Circle extends Tag {
    lazy val switch = Cross
  }

  case class Board(private val tags: Array[Tag]) {

    def play(x: Int, y: Int, tag: Tag): Board = {
      val newtags = tags.clone
      val idx = index(x, y)
      if (tags(idx) != Neither) {
        throw new RuntimeException("")
      }
      newtags(idx) = tag
      this.copy(tags = newtags)
    }

    private def index(x: Int, y: Int) = 3 * y + x

    def hasWon(tag: Tag): Boolean = {
      def eq(x: Int, y: Int): Boolean =
        tags(index(x, y)) == tag
      ((0 until 3).exists { y =>
        (0 until 3).forall(eq(_, y))
      }) ||
      ((0 until 3).exists { x => (0 until 3).forall(eq(x, _)) }) ||
      (0 until 3).forall(x => eq(x, x)) ||
      (0 until 3).forall(x => eq(x, 2 - x))
    }

    val isFull: Boolean = {
      (0 until 3).forall { y =>
        (0 until 3).forall { x => tags(index(x, y)) != Neither }
      }
    }

    def empty: Seq[(Int, Int)] = {
      (for {
        y <- 0 until 3
        x <- 0 until 3
      } yield {
        tags(index(x, y)) match {
          case Neither => Some((x, y))
          case _       => None
        }
      }).flatten
    }

    def strLines: Seq[String] = {
      for {
        y <- 0 until 3
      } yield {
        (for { x <- 0 until 3 } yield {
          tags(index(x, y)) match {
            case Cross   => "x"
            case Circle  => "o"
            case Neither => "."
          }
        }).mkString(" ")
      }
    }

    override def equals(other: Any): Boolean = {
      other match {
        case that: Board =>
          this.tags sameElements that.tags
        case _ => false
      }
    }

    override val hashCode: Int = {
      this.tags.toSeq.hashCode
    }
  }

  object Board {
    def apply(): Board = Board(Array.fill(9)(Neither))
  }

  case class Player(tag: Tag) {
    def other = Player(tag.switch)
  }

  import scappla.tensor._
  import scappla.tensor.Tensor._
  import scappla._
  import scappla.distributions.{Categorical, Bernoulli}
  import scappla.guides._
  import scappla.Functions._

  case class BoardDim(size: Int) extends Dim[BoardDim]

  case class Guide(
      initial: ArrayTensor,
      prior_param: Value[ArrayTensor, BoardDim],
      boardDim: BoardDim,
      size: Int,
      negate: Boolean
  ) {
    def prior = {
      if (negate) {
        InvCategorical(ConstantExpr(exp(prior_param)))
      } else {
        Categorical(ConstantExpr(exp(prior_param)))
      }
    }

    val posterior_param = Param(initial, boardDim)
    val posterior = if (negate) {
      BBVIGuide(InvCategorical(exp(posterior_param)))
    } else {
      BBVIGuide(Categorical(exp(posterior_param)))
    }

    def step(interpreter: Interpreter): Unit = {
      val pp_value = interpreter.eval(posterior_param)
      prior_param.dv((pp_value - prior_param).v)
    }
  }

  object Guide {
    private val guides: scala.collection.mutable.Map[Board, Guide] =
      scala.collection.mutable.Map.empty

    private val learner = new Learner(0.5)

    def size: Int = guides.size

    def apply(board: Board, tag: Tag): Guide = {
      if (!guides.contains(board)) {
        guides(board) = newGuide(board, tag == Circle)
      }
      guides(board)
    }

    private def newGuide(board: Board, negate: Boolean): Guide = {
      val free = board.empty
      // println(s"free: ${free}")
      val boardDim = BoardDim(free.size)
      val initial = ArrayTensor(boardDim.sizes, Array.fill(free.size)(0f))

      Guide(initial, learner.param(initial, boardDim), boardDim, free.size, negate)
    }
  }

  val startBoard = Board()
  // .play(1, 1, Cross)
  // .play(2, 1, Cross)
  // .play(1, 0, Circle)
  // .play(1, 2, Circle)
  val startTag = Circle
// val startTag = Cross


  val model = infer {

    def next(board: Board, player: Player): (Seq[(Int, Int)], Tag) = {
      if (board.hasWon(Cross)) {
        observe(Bernoulli(1 / 0.01), true)
        (Seq.empty, Cross)
      } else if (board.hasWon(Circle)) {
        observe(Bernoulli(0.01), true)
        (Seq.empty, Circle)
      } else if (board.isFull) {
        (Seq.empty, Neither)
      } else {
        val guide = Guide(board, player.tag)
        val options = board.empty
        val pos = sample(guide.prior, guide.posterior)
        val (x, y) = options(pos)
        val (sequence, winner) =
          next(board.play(x, y, player.tag), player.other)
        /*
        if (winner == Neither) {
          observe(Bernoulli(0.6), false)
        } else if (winner != player.tag) {
          observe(Bernoulli(0.8), false)
        }
        */
        ((x, y) +: sequence, winner)
      }
    }
    next(startBoard, Player(startTag))
  }

  import scappla.optimization._

  val board = Board()
  val optimizer = new Adam(0.01)
  val interpreter = new scappla.OptimizingInterpreter(optimizer)
  var wins = Map.empty[Tag, Int].withDefaultValue(0)
  var sequences = Map.empty[Seq[Tuple2[Int, Int]], Int].withDefaultValue(0)
  // for { iter <- 1 until 1000001 } {
  var iter = 0
  while (true) {
    interpreter.reset()
    val (sequence: Seq[Tuple2[Int, Int]], winner) = model.sample(interpreter)
    sequence.foldLeft((startBoard, startTag: Tag)) {
      case ((board, tag), (x, y)) =>
        Guide(board, tag).step(interpreter)
        (board.play(x, y, tag), tag.switch)
    }
    wins = wins + (winner -> (wins(winner) + 1))
    sequences = sequences + (sequence -> (sequences(sequence) + 1))
    if (iter % 10000 == 0) {
      println(
        s"X: ${wins(Cross) / iter.toFloat}, O: ${wins(Circle) / iter.toFloat}, -: ${wins(Neither) / iter.toFloat}"
      )
      println(s"$winner (${Guide.size} guides)")

      val currentSeq = sequence.scanLeft((startBoard, startTag: Tag)) {
        case ((board, tag), (x, y)) =>
          Guide(board, tag).step(interpreter)
          val nextBoard = board.play(x, y, tag)
          (nextBoard, tag.switch)
      }.drop(1)
      (for { y <- 0 until 3 } yield {
        (for { (board, _) <- currentSeq } yield {
          board.strLines(y)
        }).mkString("   ")
      }).foreach(println _)

      // sequence.head
      val boards = (for {
        (topSequence, _) <- sequences.toSeq.sortBy(-_._2).take(10)
      } yield {
        val (lastBoard, _) = topSequence.foldLeft((startBoard, startTag: Tag)) {
          case ((b, tag), (x, y)) => (b.play(x, y, tag), tag.switch)
        }
        lastBoard
      })
      (for { y <- 0 until 3 } yield {
        (for { board <- boards } yield {
          board.strLines(y)
        }).mkString("   ")
      }).foreach(println _)

      // lastBoard.strLines.foreach(println _)
    }
    iter += 1
  }

}

class Learner(lr: Double) extends Optimizer {

  override def param[@specialized(Float, Double) X, S](
      initial: X,
      shp: S,
      name: Option[String]
  )(implicit bf: BaseField[X, S]): Value[X, S] = {
    new Value[X, S] {

      private var value = initial

      override def field = bf

      override val shape = shp

      def v: X = value

      def dv(delta: X): Unit = {
        value = bf.plus(
          value,
          field.times(delta, bf.fromDouble(lr, shp))
        )
      }
    }
  }

}

/**
  * Categorical variant that has a mis-alignment between sampling and (log) probability.
  * It is intended for the adverserial player - the one who's chances of winning we want to 
  * minimize.
  */
case class InvCategorical[S <: Dim[_], D: TensorData](pExpr: Expr[D, S]) extends Distribution[Int] {

  private val total = sum(pExpr)

  override def sample(interpreter: Interpreter): Int = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    val draw = Random.nextDouble() * totalv.v
    val cs = cumsum(p, p.shape)
    val index = p.shape.size - count(cs, GreaterThan(draw.toFloat))
    if (index == p.shape.size) {
      // rounding error - should occur only very rarely
      p.shape.size - 1
    } else {
      index
    }
  }

  override def observe(interpreter: Interpreter, index: Int): Score = {
    val p = interpreter.eval(pExpr)
    val totalv = interpreter.eval(total)
    val value = at(p, Index[S](List(index)))
    -log(value / totalv)
  }

}
