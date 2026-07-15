# The theory of value betting

A betting event is a random experiment. Every outcome has some probability of occurring, even an unlikely one, such as more than
ten goals in a soccer match, and nobody knows what those probabilities actually are.

## Fair odds

The bookmaker estimates the probability $p$ of an outcome and offers odds $o$ on it. A bet of one unit returns $o$ if the outcome
occurs and nothing otherwise, so its expected profit is

$$
\mathbb{E}[\Pi] = p \, o - 1.
$$

The odds are fair when this is zero, that is when

$$
o = \frac{1}{p}.
$$

At fair odds, neither side makes profit in the long run.

## The bookmaker's margin

Bookmakers do not offer fair odds. They shorten them, so that the implied probability $1/o$ of every outcome is a little higher
than the probability they estimated. Across the $n$ mutually exclusive outcomes of an event, the implied probabilities therefore
sum to more than one:

$$
\sum_{i=1}^{n} \frac{1}{o_i} = 1 + m, \qquad m > 0.
$$

The excess $m$ is the over-round, and it is the bookmaker's margin. Note what has *not* changed: the bookmaker still has to
estimate $p$. The margin protects a good estimate, it does not replace one.

## Value bets

The bettor can estimate the probabilities too. Write the bettor's estimate as $\hat{p}$. A bet is a value bet when the bettor's
estimate exceeds the probability implied by the offered odds:

$$
\hat{p} > \frac{1}{o} \quad \Longleftrightarrow \quad \hat{p} \, o - 1 > 0,
$$

that is, when the bet has positive expected profit under the bettor's own estimate. Selecting value bets is the only betting
strategy that makes sense over the long run.

The caveat matters: neither side observes the true $p$. A value bet is a claim that $\hat{p}$ is closer to the truth than $1/o$ is,
and the bettor can be wrong, the bookmaker can be wrong, or both.

## Is it hopeless?

Bookmakers have more data, more computing power and teams of analysts. It is tempting to conclude that competing with them is
pointless, but that does not follow. Bookmakers balance many concerns beyond accuracy: their exposure, their competitors, the
weight of public money, which is why the odds offered on the same event vary noticeably from one bookmaker to another. That
variation is the opening.

The goal is therefore not to build an arbitrarily accurate model of football. It is to identify value bets systematically and
backtest them honestly. A realistic aim, and the one `sports-betting` is built to serve.
