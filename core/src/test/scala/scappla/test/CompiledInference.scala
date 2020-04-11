package scappla.test

import scappla._
import scappla.distributions._
import scala.collection.compat.BuildFrom

class CompiledInference {

  trait Data[T] {

    /**
      * The value of the data.  Known when used to wrap observational data,
      * in which case `observe` statements condition their distribution on it
      * to provide the likelihood.
      *
      * When compiling the model, the value is sampled (for primitive types)
      * or built from components (for compound types).  The value is only
      * known when the forward pass has completed and sufficient information
      * has been provided.
      */
    def value: T
  }

  object Data {

    def apply[T, DT <: Data[T]](
        value: T
    )(implicit df: DataFactory.Aux[T, DT]): DT =
      df(Some(value))
  }

  /**
    * Primitive data types can be sampled directly from (or observed for) elemental
    * probability distributions that are able to calculate (normalized) likelihoods.
    */
  class Primitive[T](observation: Option[T]) extends Data[T] {
    private var sample: Option[T] = None

    lazy val value: T = observation.getOrElse(sample.get)

    def set(v: T): Unit = {
      assert(observation.isEmpty && sample.isEmpty)
      sample = Some(v)
    }
  }

  type RealData = Primitive[Real]

  type BoolData = Primitive[Boolean]

  trait DataFactory[T] {

    type DataType <: Data[T]

    def apply(t: Option[T]): DataType
  }

  object DataFactory {

    def apply[T](
        implicit o: DataFactory[T]
    ): DataFactory.Aux[T, o.DataType] = o

    type Aux[T, ET0 <: Data[T]] = DataFactory[T] {
      type DataType = ET0
    }

    implicit val boolFactory: DataFactory.Aux[Boolean, BoolData] = new DataFactory[Boolean] {

      type DataType = BoolData

      def apply(b: Option[Boolean]) = new BoolData(b)
    }

    implicit val realFactory: DataFactory.Aux[Real, RealData] = new DataFactory[Real] {

      type DataType = RealData

      def apply(t: Option[Real]): RealData = new RealData(t)
    }

    implicit def listFactory[T](
        implicit edf: DataFactory[T]
    ): DataFactory.Aux[List[T], ListData[T, edf.DataType]] = new DataFactory[List[T]] {

      type DataType = ListData[T, edf.DataType]

      def apply(t: Option[List[T]]): DataType = new ListData(t)(edf)
    }

    implicit def tupleFactory[T1, T2](
        implicit
        edf1: DataFactory[T1],
        edf2: DataFactory[T2]
    ): DataFactory.Aux[(T1, T2), TupleData[T1, T2, edf1.DataType, edf2.DataType]] = new DataFactory[(T1, T2)] {

      type DataType = TupleData[T1, T2, edf1.DataType, edf2.DataType]

      def apply(v: Option[(T1, T2)]): DataType = new TupleData(v)(edf1, edf2)
    }
  }

  class ListData[T, ET <: Data[T]](
      observation: Option[List[T]]
  )(
      implicit
      elFactory: DataFactory.Aux[T, ET]
  ) extends Data[List[T]] {

    private val builder: Option[ObsBuilder] =
      if (observation.isEmpty) Some(new ObsBuilder()) else None

    lazy val value: List[T] = observation.getOrElse(builder.get.build)

    lazy val isEmpty: BoolData = {
      if (observation.isEmpty)
        builder.get.empty
      else
        new BoolData(Some(observation.get.isEmpty))
    }

    lazy val head: ET = {
      if (observation.isDefined) {
        elFactory(observation.map { _.head })
      } else {
        builder.get.head
      }
    }

    lazy val tail: ListData[T, ET] = {
      if (observation.isDefined) {
        new ListData(Some(observation.get.tail))
      } else {
        builder.get.tail
      }
    }

    private class ObsBuilder() {
      lazy val empty: BoolData = new BoolData(None)
      lazy val head: ET = elFactory(None)
      lazy val tail: ListData[T, ET] = new ListData(None)

      lazy val build: List[T] = {
        if (empty.value) {
          Nil
        } else {
          head.value :: tail.value
        }
      }
    }
  }

  class TupleData[T1, T2, ET1 <: Data[T1], ET2 <: Data[T2]](
      observation: Option[(T1, T2)]
  )(
      implicit
      elf1: DataFactory.Aux[T1, ET1],
      elf2: DataFactory.Aux[T2, ET2]
  ) extends Data[(T1, T2)] {

    private val builder: Option[Builder] =
      if (observation.isEmpty) Some(new Builder()) else None

    lazy val value: (T1, T2) =
      observation.getOrElse(builder.get.build)

    lazy val _1: ET1 = {
      if (observation.isDefined) {
        elf1(observation.map { _._1 })
      } else {
        builder.get._1
      }
    }

    lazy val _2: ET2 = {
      if (observation.isDefined) {
        elf2(observation.map { _._2 })
      } else {
        builder.get._2
      }
    }

    private class Builder {
      lazy val _1: ET1 = elf1(None)
      lazy val _2: ET2 = elf2(None)

      def build: (T1, T2) = (_1.value, _2.value)
    }
  }

  def observe[T, DT <: Primitive[T]](dist: Distribution[T], obs: DT): T = ???

  def generate(obs: ListData[Real, RealData]): List[Real] = {
    val last = observe(Bernoulli(0.5), obs.isEmpty)
    if (last) {
      Nil
    } else {
      val value = observe(Normal(0.5, 1.0), obs.head)
      value :: generate(obs.tail)
    }
  }

  val obs = Data(List(1.0: Real, 2.0: Real))
  val realData = Data(1.0: Real)
  val tupleData = Data((1.0: Real, false))

  val values = generate(obs)
}
