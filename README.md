Scala Infer
===========
With the addition of a few keywords, `scala-infer` turns scala into a probabilistic programming language.
To achieve scalability, inference is based on gradients of the (variational approximation to) the
posterior distribution.  Each draw from a distribution is accompanied by a *guide* distribution.

A probabilistic model in `scala-infer` is written as regular scala code.  Values are drawn from
distributions and are used to generate data.  Such a model is known as *generative*, as it provides
an explicit process.

Three new keywords are introduced:
* `infer` to define a model
* `sample` to draw a random variable from a distribution
* `observe` to bind data to distributions in the model

When a value is sampled, two distributions are needed; the *prior* and the (variational
approximation to) the *posterior*.  Actual sample values are drawn from the posterior, but
the prior is the starting point of the buildup of the posterior.  Without observations, the
posterior would be equal to the prior.

Parameters of the variational posterior are optimized by gradient descent.  For each sample from
the model, a backward pass calculates the gradients of the loss function.  For variational inference,
the loss function is the ELBO, a lower bound on the evidence.

Different tactics are used for discrete and continuous variables.  For continuous variables,
the reparametrization trick can be used to obtain a low variance estimator.  Discrete variables
use black-box variational inference, which only requires gradients of the score function to the
parameters.

Running the project
-------------------
While intended to become a library to be used, so far the only used ways of triggering the
macro expansion and execution is to use either
* `sbt macros/test`, or
* `sbt app/run`

Example: Sprinkler system
-------------------------
This is the rain-sprinkler-wet-grass system from the Wikipedia entry on [Bayesian
Networks](https://en.wikipedia.org/wiki/Bayesian_network).  It features a number of discrete
(boolean) random variables, an observation (the grass is wet) and an variational posterior
distribution that is optimized to approximate the exact posterior.

```scala
// optimization algorithm: (stochastic) gradient descent
val sgd = new SGD()

// posterior distribution for the sprinkler, conditional on rain.
// The parameters run over the full real axis, they are mapped to the
// domain [0,1] by the sigmoid transformation.
val inRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))
val noRain = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

// posterior distribution for the rain
val rainPost = Bernoulli(sigmoid(sgd.param(0.0, 10.0)))

// full model of the rain-sprinkler-grass system.  
val model = infer {

  // conditional sampling of the sprinkler.  The probability that
  // the sprinkler turned on, is dependent on whether it rained or not.
  val sprinkle = {
    rain: Boolean =>
      if (rain) {
        sample(Bernoulli(0.01), inRain)
      } else {
        sample(Bernoulli(0.4), noRain)
      }
  }

  val rain = sample(Bernoulli(0.2), rainPost)
  val sprinkled = sprinkle(rain)

  val p_wet = (rain, sprinkled) match {
    case (true,  true)  => 0.99
    case (false, true)  => 0.9
    case (true,  false) => 0.8
    case (false, false) => 0.001
  }

  // bind model to data / add observation
  observe(Bernoulli(p_wet), true)

  // return quantity we're interested in
  rain
}
```
The example shows a number of features:
* control flow (`if (...) ... else ...`) can be based on random variables
* it's possible to define functions of random variables

Example: linear Regression
--------------------------
Here we showcase linear regression on 2 input variables.  All variables are continuous here, with
some fixed values used to generate a data set and a model to infer these parameters from the data.
```scala
// generate data; parameters should be recovered by inference algorithm
val data = {
  val alpha = 1.0
  val beta = (1.0, 2.5)
  val sigma = 1.0

  for {_ <- 0 until 100} yield {
    val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
    val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
    (X, Y)
  }
}

// choose an optimization algorithm, gradient descent with momentum
// each parameter could have its own optimizer
val sgd = new SGDMomentum(mass = 100)

// set up variational approximation to the posterior distribution
val aPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
val b1Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
val b2Post = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))
val errPost = Normal(sgd.param(0.0, 1.0), exp(sgd.param(0.0, 1.0)))

// the actual model.  Draw variables from prior distributions and link those variables to
// the posterior approximation.
val model = infer {
  val a = sample(Normal(0.0, 1.0), aPost)
  val b1 = sample(Normal(0.0, 1.0), b1Post)
  val b2 = sample(Normal(0.0, 1.0), b2Post)
  val err = exp(sample(Normal(0.0, 1.0), errPost))

  // iterate over data points to define the observations
  data.foreach[Unit] {
    entry: ((Double, Double), Double) =>
      val ((x1, x2), y) = entry
      observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
  }

  // return the values that we're interested in
  (a, b1, b2, err)
}

// warm up - each sample of the model triggers a gradient descent step
Range(0, 1000).foreach { i =>
  sample(model)
}

// print some samples
Range(0, 10).foreach { i =>
  val l = sample(model)
  val values = (l._1.v, l._2.v, l._3.v, l._4.v)
  println(s"  $values")
}
```
Here, we not only inject the variational posterior distribution into the model, but the data as
well.  Some things to note here 
* we can naturally iterate over the data and declare observations - the used data types `Seq` and
`Tuple2` have no special meaning and neither has the `foreach` method
* while real parameters and random variables run over the whole real axis, they can be
mapped to the interval `(0, Inf)` by the `exp` function

Automatic Differentiation
-------------------------
To compute gradients, back propagation on the gradient of the (log) probability is used.  To be
able to do this, a variant of scala's native `Double` is used.  A scala-infer `Real` not only has a
value, but is also able to back-propagate gradients.  It is possible to use familiar scala syntax,
with operators like
`+`, `-`, `*` and `/`:
```scala
val x : Real = 0.0
val y : Real = 1.0
val z = x * y
```
A number of functions are available out of the box, `log`, `exp`, `pow` and `sigmoid`.  It is 
possible to combine these in custom functions by using the `toReal` macro:
```scala
val realFn = toReal { x: Double => 2 * x }
```
This should not be needed for regular model definitions, though.  These functions are typically
part of the likelihood, prior or posterior definitions.

Reparametrization Gradient
--------------------------
Continuous random variables from suitable distributions can be recast in a "reparametrized" form.
A sample is obtained by sampling a fixed-parameter distribution, followed by a deterministic
transformation specified by the parameters of the target distribution.

The variance of the gradient obtained for the parameters is reduced further by using the "Path
Derivative", as detailed in [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for
Variational Inference](https://arxiv.org/abs/1703.09194) by Roeder, Wu and Duvenaud.

Black-Box Variational Inference
-------------------------------
Dealing with discrete random variables is necessary to support a general programming language,
with control flow depending on sampled variables.  For discrete variables reparametrization is
not possible and we resort to Black-Box Variational Inference.  This suffers from large variance
on estimates of the gradient of the posterior probability with respect to the variational
parameters.

Two ways of reducing the variance are
* Rao-Blackwellization
* Control variates

The first (Rao-Blackwellization) is implemented by limiting, for each variable, the posterior to
elements in its Markov blanket.  The prior probability links the variable to its parents,
likelihoods of observations and prior probabilities of downstream variables link it to children
and their (other) parents.  This procedure eliminates irrelevant other variables as sources of
variance.

A simple control variate (moving average of the log-posterior) is used to reduce variance of
the gradient further.

For further reading, see [Black Box Variational Inference](https://arxiv.org/abs/1401.0118) by
Ranganath, Gerrish and Blei.

