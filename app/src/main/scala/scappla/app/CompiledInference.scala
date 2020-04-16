package scappla.app

import scappla._
import scappla.NoopInterpreter
import scappla.distributions._
import scala.collection.compat.BuildFrom
import scappla.distributions.Poisson

object CompiledInference extends App {

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

    def empty[T](implicit df: DataFactory[T]): T = df(None)

    type Aux[T] = Data { type Type = T }
  }

  /**
    * Primitive data types can be sampled directly from (or observed for) elemental
    * probability distributions that are able to calculate (normalized) likelihoods.
    */
  trait Primitive[T] extends Data {

    type Type = T

    def set(v: T): Unit
  }

  class PrimitiveD[T](observation: Option[T]) extends Primitive[T] {
    private var sample: Option[T] = None

    lazy val value: T = observation.getOrElse(sample.get)

    def set(v: T): Unit = {
      assert(observation.isEmpty && sample.isEmpty)
      sample = Some(v)
    }
  }

  type RealD = PrimitiveD[Real]

  type BoolD = PrimitiveD[Boolean]

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

    def size: SizeD = {
      new SizeD
    }

    class SizeD extends Primitive[Int] {

      lazy val value = {
        if (observation.isDefined) {
          observation.get.size
        } else {
          builder.get.size
        }
      }

      def set(size: Int): Unit = {
        if (size == 0) {
          builder.get.empty.set(true)
        } else {
          builder.get.empty.set(false)
          tail.size.set(size - 1)
        }
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

      def size: Int = {
        if (empty.value) {
          0
        } else {
          1 + tail.size.value
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

  def observe_train[T, DT <: Primitive[T]](dist: Distribution[T], obs: DT)(
      implicit interpreter: Interpreter
  ): T = {
    val t = dist.sample(interpreter)
    obs.set(t)
    t
  }

  def observe_test[T, DT <: Primitive[T]](dist: Distribution[T], obs: DT)(
      implicit interpreter: Interpreter
  ): T = {
    dist.observe(interpreter, obs.value)
    obs.value
  }

  /*
  def foreach[T <: Data](sizeDist: Distribution[Int], list: ListD[T])(
      fn: T => Unit
  ): Unit = {
    val size = observe(sizeDist, list.size)
    var current = list
    while (!current.isEmpty.value) {
      fn(current.head)
      current = current.tail
    }
  }
   */

  def foldLeft[T <: Data, S](sizeDist: Distribution[Int], list: ListD[T])(
      zero: S
  )(fn: (S, T) => S)(implicit interpreter: Interpreter): S = {
    val size = observe_train(sizeDist, list.size)
    var current = list
    var value = zero
    while (!current.isEmpty.value) {
      value = fn(value, current.head)
      current = current.tail
    }
    value
  }

  def generate(
      obs: ListD[RealD]
  )(implicit interpreter: Interpreter): List[Real] = {
    val last = observe_test(Bernoulli(0.5), obs.isEmpty)
    if (last) {
      Nil
    } else {
      val value = observe_test(Normal(0.5, 1.0), obs.head)
      value :: generate(obs.tail)
    }
  }

  {
    implicit val interpreter: Interpreter = NoopInterpreter
    val total = foldLeft(
      Poisson(25.0: Real),
      Data.empty[ListD[RealD]]
    )(0) {
      case (num, value) =>
        observe_train(Normal(1.0, 0.2), value)
        num + 1
    }
    println(s"Total: $total")
  }

  val obs = Data(List(1.0: Real, 2.0: Real))
  val realData = Data(1.0: Real)
  val tupleData = Data((1.0: Real, false))

  {
    implicit val interpreter: Interpreter = NoopInterpreter
    val values = generate(obs)
  }
}
