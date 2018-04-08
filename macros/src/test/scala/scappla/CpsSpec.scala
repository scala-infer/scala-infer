package scappla

import org.scalatest.FlatSpec

class CpsSpec extends FlatSpec {

  "The CPS transformation" should "transform apply arg into block" in {
    @sampled
    def toTransform(): Int = {
      sample(ConstantDistribution(1 + 2))
    }
    assert(toTransform() == 3)
  }

  it should "transform sampled arg into earlier block" in {
    @sampled
    def toTransform(): Int = {
      sample(ConstantDistribution(sample(ConstantDistribution(1))))
    }
    assert(toTransform() == 1)
  }
}
