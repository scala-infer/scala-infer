package scappla

import Functions._

object SUT {
  val square = autodiff { (z: Double) => z * z }
  val fn = autodiff { (z: Double) => z + z * square(z) }
}

object TestAutoDiff extends App {

  /*
  val bw = DValue.ad {
    (z: Double) => {
      val x = z + z * z * z
      x
    }
  }

  val z = new DVariable(2.0)
  val bwz = bw(z)
  bwz.dv(1.0)
  bwz.complete()
  assert(bwz.v == 10.0)
  assert(z.grad == 13.0)
  */

  val z = new DVariable(2.0)
  val bwz = SUT.fn(z)
  bwz.dv(1.0)
  bwz.complete()
  assert(bwz.v == 10.0)
  assert(z.grad == 13.0)

  /*
  val bla = new Function1[Double, Double] {
    def apply(x: Double) = x * x
  }
  */

/*
  val fn = autodiff {
    (z: Double) =>
      z + pow(2.0, z)
  }
*/

}
