package scappla.app

import java.io.{File, PrintWriter}

import scala.util.Random
import scala.math.{exp, log, pow}
import breeze.linalg._
import breeze.optimize.{DiffFunction, LBFGS, NaNHistory}
import breeze.stats.distributions.Gaussian

object TestLBFGS extends App {

  val hidden = (0 until 1000).scanLeft(0.0) { case (v, _) =>
    v + Random.nextGaussian() * 0.1
  }
  val data = hidden.map { v =>
    (v, v + 0.2 * Random.nextGaussian())
  }

  val x = data.map { _._2 }
  val distWeights = (for {
    i <- x.indices
    j <- x.indices
  } yield {
    val dist = if (i < j)
      j - i
    else
      i - j

    (dist, x(i) * x(j))
  }).groupBy(_._1).map {
    case (dist, pairs) =>
      (dist: Double, pairs.map { _._2 }.sum)
  }
  val diag = x.map { xi => xi * xi }.sum.toDouble

  val N: Double = x.size

  def f(lambda: Double, s_x: Double) = {
    val g = (1f - lambda) / (1f + lambda)
    val L = -(diag - g * distWeights.map {
      case (dist, weight) =>
        pow(lambda, dist) * weight
    }.reduce {
      _ + _
    }) / (2f * s_x * s_x) + N * log(lambda) / 2f - N * log(s_x)
    L
  }

  def dfdl(lambda: Double, s_x: Double) = {
    val g = (1f - lambda) / (1f + lambda)
    g * distWeights.map {
      case (dist, weight) => dist * pow(lambda, dist - 1) * weight
    }.sum / (2f * s_x * s_x) +
      (-1f / (1f + lambda) - (1f - lambda) / ((1f + lambda) * (1f + lambda))) * distWeights.map {
      case (dist, weight) => pow(lambda, dist) * weight
    }.sum / (2f * s_x * s_x) +
      N / (2f *  lambda)
  }

  def dfdsx(lambda: Double, s_x: Double) = {
    val g = (1f - lambda) / (1f + lambda)
    (diag - g * distWeights.map {
      case (dist, weight) =>
        pow(lambda, dist) * weight
    }.sum) / (s_x * s_x * s_x) - N / s_x
  }

  val initial = (0 until 20).map { _ =>
    val x_0 = 0.1 * Random.nextGaussian()
    val x_1 = 0.1 * Random.nextGaussian() - 2

    val lambda = 1f / (1f + exp(-x_0))
    val s_x = exp(x_1)
    val x = DenseVector(lambda, s_x)

//    println(s"Lambda: ${lambda}, S_X: ${s_x}")

    val dfdl_grad = dfdl(lambda, s_x)
//    val dfdl_diff = (f(lambda + 0.001, s_x) - f(lambda, s_x)) / 0.001
//    println(s"Lambda grad: ${dfdl_grad}, diff: ${dfdl_diff}")
    val g_l = lambda * (1 - lambda) * dfdl_grad

    val dfdsx_grad = dfdsx(lambda, s_x)
//    val dfdsx_diff = (f(lambda, s_x + 0.001) - f(lambda, s_x)) / 0.001
//    println(s"S_X grad: ${dfdsx_grad}, diff: ${dfdsx_diff}")
    val g_sx = s_x * dfdsx(lambda, s_x)
    val g = DenseVector(g_l, g_sx)

    (x, g)
  }.toList

  // val optimizer = new SGFS(0.5, 0.9, 0.1)
  // val optimizer = new MyLGBFS(0.5, 0.9, 0.1)
//  val optimizer = new MyAdam(0.5, 0.9, 0.1)
  // val optimizer = new DirectLGBFS(0.5, 0.9)
  for {
    (name, optimizer) <- List(
      ("adam", new MyAdam(0.5, 0.9, 0.1)),
      ("direct", new DirectLGBFS(0.5, 0.9)),
      ("sgfs", new SGFS(0.5, 0.9, 0.1))
    )
  } {
    val out = new File(s"/tmp/path-$name.csv")
    val writer = new PrintWriter(out)
    val outcome = (0 until 200).foldLeft(
      optimizer.create(initial.map {
        _._1
      }, initial.map {
        _._2
      })
    ) {
      case (state, iter) =>
        val next_x = optimizer.sample(state)
        val x_0 = next_x(0)
        val x_1 = next_x(1)
        val lambda = 1f / (1f + exp(-x_0))
        val s_x = exp(x_1)

        val g_l = lambda * (1 - lambda) * dfdl(lambda, s_x)
        val g_sx = s_x * dfdsx(lambda, s_x)
        val g = DenseVector(g_l, g_sx)

        val s_z = s_x * math.sqrt(lambda + 1.0 / lambda - 2.0)
        writer.println(s"$lambda, $s_x, $s_z, $g_l, $g_sx")
        println(s"$iter: l: ${lambda}, s_x: ${s_x}, s_z: ${s_z}")

        optimizer.update(state, next_x, g)
    }
    writer.close()
  }
}

case class State(
    g: DenseVector[Double],
    x: DenseVector[Double],
    gg: DenseMatrix[Double],
    xx: DenseMatrix[Double],
    gx: DenseMatrix[Double]
)

trait QuasiNewtonOptimizer {

  def beta1: Double

  def beta2: Double

  def create(xs: List[DenseVector[Double]], gs: List[DenseVector[Double]]): State = {
    val N = xs.size.toDouble
    val n = xs.head.size

    val g = gs.reduce { _ + _ } / N
    val x = xs.reduce { _ + _ } / N

    val xx = DenseMatrix.fill[Double](n, n)(0.0)
    for { xi <- xs } {
      xx += (xi - x) * (xi - x).t / (N - 1)
    }
    val gx = DenseMatrix.fill[Double](n, n)(0.0)
    for { (gi, xi) <- gs.zip(xs) } {
      gx += (gi - g) * (xi - x).t / (N - 1)
    }
    val gg = DenseMatrix.fill[Double](n, n)(0.0)
    for { gi <- gs } {
      gg += (gi - g) * (gi - g).t / (N - 1)
    }
    State(g, x, gg, xx, gx)
  }

  def update(state: State, x: DenseVector[Double], g: DenseVector[Double]): State = {
    val g_avg = beta1 * state.g + (1 - beta1) * g
    val x_avg = beta1 * state.x + (1 - beta1) * x
    State(
      g_avg,
      x_avg,
      beta2 * state.gg + (1 - beta2) * (g - g_avg) * (g - g_avg).t,
      beta2 * state.xx + (1 - beta2) * (x - x_avg) * (x - x_avg).t,
      beta2 * state.gx + (1 - beta2) * (g - g_avg) * (x - x_avg).t
    )
  }

  def sample(state: State): DenseVector[Double]
}

class MyAdam(val beta1: Double, val beta2: Double, val alpha: Double) extends QuasiNewtonOptimizer {

  def sample(state: State): DenseVector[Double] = {
    val diag_gg = breeze.linalg.diag(state.gg)
    state.x + alpha * state.g /:/ (breeze.numerics.sqrt(diag_gg) + 0.01)
  }
}

class SGFS(val beta1: Double, val beta2: Double, noise: Double) extends QuasiNewtonOptimizer {

  def sample(state: State): DenseVector[Double] = {
    val n = state.x.size
    val gg_inv = state.gg \ DenseMatrix.eye[Double](n)

    // simulate noise coming from measurement
    // we use the exact posterior, whereas normally there is noise from sampling
    val e = DenseVector.rand[Double](n, Gaussian(0.0, 1.0))
    val gg_chol = cholesky(state.gg)
    val dg = gg_chol * e * noise

    val dx = gg_inv * (state.g + dg)
    state.x + dx
  }
}

class DirectLGBFS(val beta1: Double, val beta2: Double) extends QuasiNewtonOptimizer {

  override def sample(state: State): DenseVector[Double] = {
    // g s_x^-1 * y = g
    // s_x^-1 * y = dx
    val y = state.gx \ state.g
    val dx = state.xx * y

    val rel_dist = dx.t * (state.xx \ dx)
    // val err_dist = err_x.t * xx_inv * err_x * N
    // println(s"  dist: ${rel_dist} ($err_dist)")
    println(s"  dist: ${rel_dist}")
    val result = state.x - dx / (1 + math.sqrt(rel_dist) / 5)
    if (result.exists {
      _.isNaN
    })
      assert(false)
    result
  }
}

class MyLGBFS(val beta1: Double, val beta2: Double, noise: Double) extends QuasiNewtonOptimizer {

  def sample(state: State): DenseVector[Double] = {

    val xx_inv = pinv(state.xx)
    val gx_inv = pinv(state.gx)

    // g s_x^-1 * y = g
    // s_x^-1 * y = dx
    val y = gx_inv * state.g
    val dx = state.xx * y

    val rel_dist = dx.t * xx_inv * dx
    // val err_dist = err_x.t * xx_inv * err_x * N
    // println(s"  dist: ${rel_dist} ($err_dist)")
    println(s"  dist: ${rel_dist}")
    val result = state.x - dx / (1 + math.sqrt(rel_dist) / 5)
    if (result.exists { _.isNaN })
      assert(false)
    result
//    x - 0.2 *  dx
  }
    /*
    val g = gs.zipWithIndex.map {
      case (gi, idx) => gi
    }.reduce { _ + _ } / N
    val x = xs.zipWithIndex.map {
      case (gi, idx) => weight(idx) * gi
    }.reduce { _ + _ } / N

    // solve <gx> * y = <g> for y
    // x = <xx> * y
    val xx = DenseMatrix.fill[Double](n, n)(0.0)
    for { (xi, idx) <- xs.zipWithIndex } {
      for { i <- 0 until n; j <- 0 until n } {
        xx(i, j) += weight(idx) * (xi(i) - x(i)) * (xi(j) - x(j)) / (N - 1)
      }
    }
    val xg = DenseMatrix.fill[Double](n, n)(0.0)
    for { ((gi, xi), idx) <- gs.zip(xs).zipWithIndex } {
      for { i <- 0 until n; j <- 0 until n } {
//        xg(i, j) += (xi(i) - x(i)) * (gi(j) - g(j))
        xg(j, i) += weight(idx) * (xi(i) - x(i)) * (gi(j) - g(j)) / (N - 1)
//        xg(i, j) += (xi(i) - x(i)) * gi(j)
      }
    }
    val gg = DenseMatrix.fill[Double](n, n)(0.0)
    for { (gi, idx) <- gs.zipWithIndex } {
      for {i <- 0 until n; j <- 0 until n } {
        gg(i, j) += weight(idx) * (gi(i) - g(i)) * (gi(j) - g(j)) / (N - 1)
//        gg(i, j) += weight(idx) * (gi(i) - 0.9 * g(i)) * (gi(j) - 0.9 * g(j))
//        gg(i, j) += weight(idx) * gi(i) * gi(j)
      }
    }
    */

    /*
    // sample a displacement
//    val B = xg.t \ xx  //  xg.t * B = xx => B is inverse Hessian
    val e = DenseVector.rand[Double](n, Gaussian(0.0, 1.0))
    val gg_inv = gg \ DenseMatrix.eye[Double](n)

    {
      val ggn = gg / N
      println("%.03f %.03f %.03f %.03f".format(ggn(0, 0), ggn(0, 1), ggn(1, 0), ggn(1, 1)))
      val ggin = N * gg_inv
      println("%.03f %.03f %.03f %.03f".format(ggin(0, 0), ggin(0, 1), ggin(1, 0), ggin(1, 1)))
    }

    val xx_chol = cholesky(xx / (N + 1))
    val eta_x = xx_chol * e
    val dg = gg * eta_x / N

//    val epsilon = 100.0
//    val gg_chol = cholesky(gg / (N * epsilon))
//    val dg = gg_chol * e
//    val dx = 2 * N * gg_inv * (g + dg) / (0.5 + 1.0 / epsilon)
//    val dx =  N * gg_inv * g
    val dx =  N * gg_inv * (g + dg)

    val rel_dist = dx.t * xx_inv * dx * N
    if (rel_dist > 1000)
      println(s"  dist: ${rel_dist} ($N)")
    x + dx / (1 + math.sqrt(rel_dist / N) / 5)
    */

}
