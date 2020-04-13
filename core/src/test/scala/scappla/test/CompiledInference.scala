package scappla.test

import scappla._
import scappla.distributions._
import scala.collection.compat.BuildFrom

class CompiledInference {

  trait Data {

    type Type

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
    def value: Type
  }

  object Data {

    def apply[T, DT <: Data.Aux[T]](
        value: T
    )(implicit df: DataFactory.Aux[T, DT]): DT =
      df(Some(value))

    type Aux[T] = Data { type Type = T }
  }

  /**
    * Primitive data types can be sampled directly from (or observed for) elemental
    * probability distributions that are able to calculate (normalized) likelihoods.
    */
  class Primitive[T](observation: Option[T]) extends Data {
    private var sample: Option[T] = None

    type Type = T

    lazy val value: T = observation.getOrElse(sample.get)

    def set(v: T): Unit = {
      assert(observation.isEmpty && sample.isEmpty)
      sample = Some(v)
    }
  }

  type RealD = Primitive[Real]

  type BoolD = Primitive[Boolean]

  trait DataFactory[DT] {

    type Type

    def apply(t: Option[Type]): DT
  }

  object DataFactory {

    def apply[T, DT <: Data.Aux[T]](
        implicit o: DataFactory.Aux[T, DT]
    ): DataFactory.Aux[T, DT] = o

    type Aux[T, DT] = DataFactory[DT] { type Type = T }

    implicit val boolFactory: DataFactory.Aux[Boolean, BoolD] =
      new DataFactory[BoolD] {

        type Type = Boolean

        def apply(b: Option[Boolean]) = new BoolD(b)
      }

    implicit val realFactory: DataFactory.Aux[Real, RealD] =
      new DataFactory[RealD] {

        type Type = Real

        def apply(t: Option[Real]): RealD = new RealD(t)
      }

    implicit def listFactory[T, DT <: Data.Aux[T]](
        implicit edf: DataFactory.Aux[T, DT]
    ): DataFactory.Aux[List[T], ListD[DT] { type Type = List[T] }] =
      new DataFactory[ListD[DT] { type Type = List[T] }] {

        type Type = List[T]

        def apply(t: Option[List[T]]) = new ListD[DT](t)(edf)
      }

    implicit def tupleFactory[T1, T2, DT1 <: Data.Aux[T1], DT2 <: Data.Aux[T2]](
        implicit
        edf1: DataFactory.Aux[T1, DT1],
        edf2: DataFactory.Aux[T2, DT2]
    ): DataFactory.Aux[(T1, T2), TupleD[DT1, DT2] { type Type = (T1, T2) }] =
      new DataFactory[TupleD[DT1, DT2] { type Type = (T1, T2) }] {

        type Type = (T1, T2)

        def apply(v: Option[(T1, T2)]) = new TupleD[DT1, DT2](v)(edf1, edf2)
      }
  }

  class ListD[DT <: Data](
      observation: Option[List[DT#Type]]
  )(
      implicit
      elFactory: DataFactory.Aux[DT#Type, DT]
  ) extends Data {

    private val builder: Option[ObsBuilder] =
      if (observation.isEmpty) Some(new ObsBuilder()) else None

    type Type = List[DT#Type]

    lazy val value: List[DT#Type] = observation.getOrElse(builder.get.build)

    lazy val isEmpty: BoolD = {
      if (observation.isEmpty)
        builder.get.empty
      else
        new BoolD(Some(observation.get.isEmpty))
    }

    lazy val head: DT = {
      if (observation.isDefined) {
        elFactory(observation.map { _.head })
      } else {
        builder.get.head
      }
    }

    lazy val tail: ListD[DT] = {
      if (observation.isDefined) {
        new ListD(Some(observation.get.tail))
      } else {
        builder.get.tail
      }
    }

    private class ObsBuilder() {
      lazy val empty: BoolD = new BoolD(None)
      lazy val head: DT = elFactory(None)
      lazy val tail: ListD[DT] = new ListD(None)

      lazy val build: List[DT#Type] = {
        if (empty.value) {
          Nil
        } else {
          head.value :: tail.value
        }
      }
    }
  }

  class TupleD[DT1 <: Data, DT2 <: Data](
      observation: Option[(DT1#Type, DT2#Type)]
  )(
      implicit
      elf1: DataFactory.Aux[DT1#Type, DT1],
      elf2: DataFactory.Aux[DT2#Type, DT2]
  ) extends Data {

    private val builder: Option[Builder] =
      if (observation.isEmpty) Some(new Builder()) else None

    type Type = (DT1#Type, DT2#Type)

    lazy val value: (DT1#Type, DT2#Type) =
      observation.getOrElse(builder.get.build)

    lazy val _1: DT1 = {
      if (observation.isDefined) {
        elf1(observation.map { _._1 })
      } else {
        builder.get._1
      }
    }

    lazy val _2: DT2 = {
      if (observation.isDefined) {
        elf2(observation.map { _._2 })
      } else {
        builder.get._2
      }
    }

    private class Builder {
      lazy val _1: DT1 = elf1(None)
      lazy val _2: DT2 = elf2(None)

      def build: (DT1#Type, DT2#Type) = (_1.value, _2.value)
    }
  }

  def observe[T, DT <: Primitive[T]](dist: Distribution[T], obs: DT): T = ???

  def generate(obs: ListD[RealD]): List[Real] = {
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
