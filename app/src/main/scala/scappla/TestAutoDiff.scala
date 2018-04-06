package scappla

object TestAutoDiff extends App {

  // c = fn(a, b)
  // ((a, b), dc) => (da, db)
  @differentiate
  def bw(z: Double): Double = z + z * z * z

  assert(bw.grad(1.0, 1.0) == 4.0)
  assert(bw.grad(0.5, 1.0) == 1.75)

  //    val bw: (Double, Double) => Double =
  //    Macro.backward[Double, Double](z => z + z * z * z)

  /*
  val bwblock = Macro.backward[Double, Double] { z =>
      val x = {
        val y = z * z
        y
      }
      x
    }
    */

}
