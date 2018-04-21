package scappla

import scappla.DValue.{ad, pow}

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

  val fn = ad { (z: Double) =>
    z + pow(2.0, z)
  }

}
