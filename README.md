Scala Infer
===========
With the addition of a few keywords, `scala-infer` turns scala into a probabilistic programming language.
To achieve scalability, inference is based on gradients of the (variational approximation to) the
posterior distribution.  Each draw from a distribution is accompanied by a *guide* distribution.

A probabilistic model in `scala-infer` is written as regular scala code.  Values are drawn from
distributions and are used to generate data.  Such a model is known as *generative*, as it provides
an explicit process.

Three new keywords are introduced:
* infer
* sample
* observe

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

Automatic Differentiation
-------------------------
To compute gradients, back propagation on the (log) probability is used.  To be able to do this, a
variant of scala's native `Double` is used.  A scala-infer `Real` not only has a value, but is also
able to back-propagate gradients.  It is possible to use familiar scala syntax, with operators like
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

For further reading, see [https://arxiv.org/abs/1401.0118]("Black Box Variational Inference") by
Ranganath, Gerrish and Blei.
