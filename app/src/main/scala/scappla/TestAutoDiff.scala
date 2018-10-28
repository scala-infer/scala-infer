package scappla

object TestAutoDiff extends App {

  import autodiff.toReal

  val bw = toReal { z: Double =>
    val x = z + z * z * z
    x
  }

  val z = new DVariable(2.0)
  val bwz = bw(z)
  bwz.dv(1.0)
  assert(bwz.v == 10.0)
  assert(z.grad == 13.0)

}
