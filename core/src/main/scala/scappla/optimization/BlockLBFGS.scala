package scappla.optimization

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.eigSym.DenseEigSym
import breeze.linalg.{DenseMatrix, DenseVector, sum, svd}

import scappla.{BaseField, Value}

// Shared state across parameters
// Is updated with each parameter sample (gradient backprop)
case class CollectState(
    g_gram: DenseMatrix[Double], // (row, col): g_row dot g_col
    x_gram: DenseMatrix[Double], // (row, col): x_row dot x_col
    xg: DenseMatrix[Double],     // (row, col): g_row dot x_col
)

// Shared state needed to calculated new position vectors
case class StepState(
    dx: DenseVector[Double],
    dg: DenseVector[Double],
)

// State per parameter that tracks the historic positions and gradients
case class ParamState[X](
    xs: List[X],
    gs: List[X]
)

class BlockLBFGS(histSize: Int = 5, learningRate: Double = 0.5) extends Optimizer with LazyLogging {

  import BlockLBFGS._

  private var iteration = 0

  private var stepState = StepState(
    DenseVector.fill(0)(0.0),
    DenseVector.fill(0)(0.0),
  )

  private var collectState = CollectState(
    DenseMatrix.fill(1, 1)(0.0),
    DenseMatrix.fill(1, 1)(0.0),
    DenseMatrix.fill(1, 1)(0.0),
  )

  override def step(): Unit = {
    val prevSize = nextSize.toDouble
    iteration += 1

    val (g_nz_ev, g_proj) = project_g(collectState.g_gram)
    val (x_nz_ev, dx_proj) = project_x(collectState.x_gram)
    // val dxg = project_xg(collectState.xg)
    // println(s" nnz_x: ${x_nz_ev.length}, nnz_g: ${g_nz_ev.length}")

    val g_proj_avg = breeze.linalg.sum(g_proj(breeze.linalg.*, ::)) / prevSize
    // dim: 1
    val dg_proj = g_proj(::, breeze.linalg.*) - g_proj_avg
    logger.debug(s"  g_proj_avg: ${g_proj_avg}")

    // val x_proj_avg = breeze.linalg.sum(x_proj(breeze.linalg.*, ::)) / prevSize
    // dim: 1
    // val dx_proj = x_proj(::, breeze.linalg.*) - x_proj_avg
    // logger.debug(s"  x_proj_avg: ${x_proj_avg}")

    // dim: xx
    val Q = dg_proj * dx_proj.t
    logger.debug(s"  Q: ${Q.rows}, ${Q.cols}")
    logger.debug(Q.toString())

    // dim: 1 / xx
    val delta_x = if (iteration > 1) {
      val QI = pinv(Q, 0.01)
      logger.debug(s"  QI: ${QI.rows}, ${QI.cols}")
      pinv(Q, 0.01) * g_proj_avg
    } else {
      DenseVector.fill(x_nz_ev.length)(0.0)
    }

    // (x_avg - x_0) in terms of coefficients of the position history
    // dim: 1
    logger.debug(s"  delta_x: ${dx_proj.t * delta_x}")

    // dim: 1
    val remaining = g_proj_avg - Q * delta_x

    // dim: xx / gg
    /*
    val scale = if (iteration > 1) {
      math.abs(
        breeze.linalg.sum(
          breeze.linalg.diag(dg_proj * dxg * dg_proj.t) /:/ g_nz_ev
        )
      )
    } else {
      0.1
    }
    */
    // val scale: Double = if (iteration > 1)
      // math.abs(breeze.linalg.sum(Q) / g_trace)
    // else
      // 0.001 / collectState.g_gram(0, 0)
    // val scale = 0.1

    // println(s"  scale: $scale")
    // logger.debug(s"  scale: $scale")

    // dim: x / g
    // val delta_g = scale * remaining
    val delta_g = remaining
    logger.debug(s"  delta_g: ${g_proj.t * delta_g}")

    // make sure we don't go absurdly far outside known territory
    val rescale = {
      val dist_x = delta_x.t * delta_x
      // val dist_g = delta_g.t * (g_nz_ev *:* delta_g)
      // 1.0 / (1.0 + math.sqrt(dist_x + dist_g) / 5)
      1.0 / (1.0 + math.sqrt(dist_x) / 5)
    }
    // println(s"  rescale: $rescale")

    // dim: x / g
    // val delta_orig = rescale * g_proj.t * total_proj
    //    logger.debug(delta_orig)

    stepState = StepState(
      rescale * dx_proj.t * delta_x,
      rescale * g_proj.t * delta_g
    )

    val next_g_gram = DenseMatrix.fill(nextSize, nextSize)(0.0)
    val next_x_gram = DenseMatrix.fill(nextSize, nextSize)(0.0)
    val next_xg = DenseMatrix.fill(nextSize, nextSize)(0.0)
    for {
      i <- 1 until nextSize
      j <- 1 until nextSize
    } {
      next_g_gram(i, j) = collectState.g_gram(i - 1, j - 1)
      next_x_gram(i, j) = collectState.x_gram(i - 1, j - 1)
      next_xg(i, j) = collectState.xg(i - 1, j - 1)
    }
    collectState = CollectState(
      g_gram = next_g_gram,
      x_gram = next_x_gram,
      xg = next_xg
    )
  }

  private def pinv(v: DenseMatrix[Double], min_size: Double): DenseMatrix[Double] = {
    val svd.SVD(s, svs, d) = svd(v)
    val trace = sum(svs)
    val vi = svs.map { v =>
      if (math.abs(v) <= min_size * trace) 0.0f else 1 / v
    }

    val svDiag = DenseMatrix.tabulate(s.cols, d.rows) { (i, j) =>
      if (i == j && i < math.min(s.cols, d.rows)) vi(i)
      else 0.0f
    }
    val res = s * svDiag * d
    res.t
  }

  def nextSize = if (iteration >= histSize) histSize else iteration + 1

  def param[X, S](initial: X, shp: S, name: Option[String])(implicit base: BaseField[X, S]): Value[X, S] = {
    new Value[X, S] {

      var state = ParamState[X](List(initial), List.empty)

      val field = base

      val shape = shp

      private var curV: Option[X] = Some(initial)

      def v: X = {
        if (curV.isEmpty) {
          val dummy = newV()
          //          logger.debug(s"  newv: ${dummy}")
          //          val nv: X = Random.nextGaussian().asInstanceOf[X]
          val nv = dummy
          curV = Some(nv)
          state = state.copy(
            xs = (nv +: state.xs).take(histSize)
          )
        }
        curV.get
      }

      private def newV(): X = {
        val x_avg = field.div(
          state.xs.reduce(field.plus),
          field.fromInt(state.xs.size, shp)
        )

        /*
        logger.debug(s"DX: ${state.xs.size}")
        logger.debug(stepState.dx)
        logger.debug(s"DG: ${state.gs.size}")
        logger.debug(stepState.dg)
        */

        // delta of x_0 - x_avg
        val dx = (for {
          i <- state.gs.indices
        } yield {
          field.plus(
            field.times(
              field.minus(state.xs(i), x_avg),
              field.fromDouble(-stepState.dx(i), shp)
            ),
            field.times(
              state.gs(i),
              field.fromDouble(-learningRate * stepState.dg(i), shp)
            )
          )
        }).reduce(field.plus)
        logger.debug(s"  X_0 - X_avg: $dx")

        val dx_head = field.plus(
          dx,
          field.minus(x_avg, state.xs.head)
        )
        logger.debug(s"  dX_head: $dx_head")

        // move head a little bit into the direction of (the expected) x_0
        field.plus(
          state.xs.head,
          dx_head
          // field.times(dx_head, field.fromDouble(learningRate, shp))
        )
      }

      def dv(dv: X): Unit = {
        val min_dv = base.negate(dv)
        val next_gs = (min_dv +: state.gs).take(histSize)
        for {
          i <- 1 until nextSize
        } {
          val g_in = base.sumAll(base.times(min_dv, next_gs(i)))
          collectState.g_gram(0, i) += g_in
          collectState.g_gram(i, 0) += g_in

          val x_in = base.sumAll(base.times(v, state.xs(i)))
          collectState.x_gram(0, i) += x_in
          collectState.x_gram(i, 0) += x_in

          collectState.xg(0, i) = base.sumAll(base.times(min_dv, state.xs(i)))
          collectState.xg(i, 0) = base.sumAll(base.times(next_gs(i), v))
        }
        collectState.g_gram(0, 0) += base.sumAll(base.times(min_dv, min_dv))
        collectState.x_gram(0, 0) += base.sumAll(base.times(v, v))
        collectState.xg(0, 0) += base.sumAll(base.times(min_dv, v))

        logger.debug(s"X: $v, DX: $dv")

        // update gradients in parameter state
        state = state.copy(
          gs = next_gs
        )
        curV = None
      }
    }
  }
}

object BlockLBFGS {

  def project_g(g_gram: DenseMatrix[Double]): (DenseVector[Double], DenseMatrix[Double]) = {
    // find independent vectors to build a projected space
    // eigenvectors:
    val size = g_gram.rows
    val g_eig: DenseEigSym = breeze.linalg.eigSym(g_gram)
    val g_trace = breeze.linalg.sum(g_eig.eigenvalues)

    val g_inv_eig_indices = g_eig.eigenvalues.toArray
        .zipWithIndex
        .filter { vi =>
          math.abs(vi._1 / g_trace) > (1e-3 / size)
        }
        .map {
          _._2
        }
        .toList
    // dim: 1 / gg
    (
      g_eig.eigenvalues(g_inv_eig_indices).toDenseVector,
    // dim: 1
      g_eig.eigenvectors.t(g_inv_eig_indices, ::).toDenseMatrix
    )
  }

  def project_x(x_gram: DenseMatrix[Double]): (DenseVector[Double], DenseMatrix[Double]) = {
    val size = x_gram.rows
    val xx_avg = breeze.linalg.sum(x_gram) / (x_gram.rows * x_gram.cols)
    val x_eig = breeze.linalg.eigSym(x_gram - xx_avg)
    val x_trace = breeze.linalg.sum(x_eig.eigenvalues)

    val x_inv_eig_indices = x_eig.eigenvalues.toArray
        .zipWithIndex
        .filter { vi =>
          // (vi._1 / x_trace) > 0.0
          (vi._1 / x_trace) > (1e-3 / size)
        }
        .map {
          _._2
        }
        .toList
    (
      x_eig.eigenvalues(x_inv_eig_indices).toDenseVector,
      x_eig.eigenvectors.t(x_inv_eig_indices, ::).toDenseMatrix
    )
  }

  def project_xg(xg: DenseMatrix[Double]): DenseMatrix[Double] = {
    val size = xg.cols.toDouble

    val xg_avg = breeze.linalg.sum(xg(breeze.linalg.*, ::)) / size
    xg(::, breeze.linalg.*) - xg_avg
  }

}