# Derivation of the updates

This note derives every formula implemented in [`src/admm_svm/solvers.py`](../src/admm_svm/solvers.py).

## Notation

The model is $x = [w;\, c] \in \mathbb{R}^{p}$ with $p = d + 1$. Stack the samples into
the *signed design matrix* $B \in \mathbb{R}^{N \times p}$ whose $i$-th row is
$y_i\,[a_i^\top,\; 1]$, so that the margin of sample $i$ is $(Bx)_i$. Let
$R = \mathrm{diag}(1, \dots, 1, 0)$, which leaves the intercept out of the regulariser.
The problem is

$$
\min_x \; f(x) = \frac{1}{N}\sum_{i=1}^{N} h\bigl((Bx)_i\bigr) + \frac{\lambda}{2}\, x^\top R\, x,
\qquad h(s) = \max(0,\, 1 - s).
$$

## 1. Subgradient descent

$h$ is differentiable everywhere except at $s = 1$. A valid subgradient of $f$ is obtained
by taking slope $-1$ where the margin is below 1 and slope $0$ elsewhere:

$$
g(x) = -\frac{1}{N}\sum_{i\,:\,(Bx)_i < 1} B_i^\top + \lambda R\, x .
$$

The iteration is $x_{k+1} = x_k - \mu_k\, g(x_k)$. With a constant step the method only
reaches a neighbourhood of the optimum whose size is proportional to the step; with
$\mu_k = \mu_0/\sqrt{k+1}$ it converges, at rate $O(1/\sqrt{k})$. The objective is not
monotone, so the best iterate is kept.

## 2. Centralized ADMM

Introduce one auxiliary variable per margin, $s = Bx$:

$$
\min_{x,\,s}\; \frac{1}{N}\sum_i h(s_i) + \frac{\lambda}{2} x^\top R x
\quad\text{subject to}\quad Bx - s = 0 .
$$

The loss is an average over $N$ samples, so the penalty is scaled the same way. With
the scaled dual variable $u$, the augmented Lagrangian is

$$
L_\rho(x, s, u) = \frac{1}{N}\sum_i h(s_i) + \frac{\lambda}{2} x^\top R x
+ \frac{\rho}{2N}\,\lVert Bx - s + u \rVert^2 - \frac{\rho}{2N}\lVert u \rVert^2 .
$$

ADMM minimises over $x$, then over $s$, then updates $u$.

**x-update.** $L_\rho$ is a quadratic in $x$. Setting its gradient to zero:

$$
\Bigl(\lambda R + \frac{\rho}{N} B^\top B\Bigr)\, x_{k+1} = \frac{\rho}{N}\, B^\top (s_k - u_k).
$$

The matrix is $p \times p$, positive definite, and does not change between iterations,
so its Cholesky factor is computed once.

**s-update.** The problem separates over samples. With $q = (Bx_{k+1})_i + (u_k)_i$,

$$
s_i = \arg\min_s \; \frac{1}{N} h(s) + \frac{\rho}{2N}(s - q)^2
    = \arg\min_s \; \frac{1}{\rho}\, h(s) + \frac{1}{2}(s - q)^2
    = \mathrm{prox}_{h/\rho}(q).
$$

Checking the three pieces of $h$ gives a closed form with $t = 1/\rho$:

$$
\mathrm{prox}_{t h}(q) =
\begin{cases}
q, & q \ge 1 & \text{(already outside the margin: leave it)}\\
1, & 1 - t \le q < 1 & \text{(close: move it exactly onto the margin)}\\
q + t, & q < 1 - t & \text{(far: push it by } t\text{)}
\end{cases}
$$

or, in one line, `where(q >= 1, q, minimum(q + t, 1))`.

**u-update.** $u_{k+1} = u_k + Bx_{k+1} - s_{k+1}$.

This is two-block ADMM on a convex problem, so it converges for every $\rho > 0$.

## 3. Distributed consensus ADMM

Node $n$ of a connected network holds the rows $B_n$ of $B$. Give each node its own
model $x_n$ and its own share of the objective,

$$
f_n(x_n) = \frac{1}{N}\sum_{i \in \text{node } n} h\bigl((B_n x_n)_i\bigr) + \frac{\lambda}{2 M}\, x_n^\top R\, x_n,
$$

where $M$ is the number of nodes, so that $\sum_n f_n(x) = f(x)$. The global problem is
equivalent to

$$
\min_{x_1, \dots, x_M} \; \sum_n f_n(x_n) \quad\text{subject to}\quad x_i = x_j \;\text{ for every link } (i, j),
$$

because on a connected network, agreement along every link implies agreement everywhere.

To let the two ends of a link update independently, add one variable $z_e$ per link
$e = (i, j)$ and require $x_i = z_e$ and $x_j = z_e$. As before, add local margins
$s_n = B_n x_n$. The constraints are

$$
B_n x_n = s_n \;\; (\text{dual } u_n), \qquad x_n = z_e \;\; (\text{dual } w_{n,e}) \;\text{ for every link } e \text{ touching } n,
$$

with penalties $\rho/N$ and $\rho_c$. The variables form two blocks, $(x_n)$ and
$(s_n, z_e)$, and every constraint ties one variable from each block. This is again
two-block ADMM, so it converges for all positive $\rho$, $\rho_c$. Within each block the
problem separates, which is what makes the algorithm distributed.

**x-update (at each node, in parallel).** Let $\mathcal{E}_n$ be the links of node $n$
and $\deg_n$ their number.

$$
\Bigl(\frac{\lambda}{M} R + \frac{\rho}{N} B_n^\top B_n + \rho_c \deg_n I\Bigr)\, x_n
= \frac{\rho}{N} B_n^\top (s_n - u_n) + \rho_c \sum_{e \in \mathcal{E}_n} (z_e - w_{n,e}).
$$

The first term on the right pulls the model towards fitting the local data, the second
towards what was agreed with the neighbours. Each node factorises its matrix once.

**s-update (local).** As in the centralized case, $s_n = \mathrm{prox}_{h/\rho}(B_n x_n + u_n)$.

**z-update (one per link).** Minimising
$\lVert x_i - z + w_{i,e}\rVert^2 + \lVert x_j - z + w_{j,e}\rVert^2$ over $z$ gives the average

$$
z_e = \tfrac{1}{2}\bigl(x_i + w_{i,e} + x_j + w_{j,e}\bigr).
$$

Computing it requires node $i$ and node $j$ to exchange $p$ numbers each.

**Dual updates.** $u_n \leftarrow u_n + B_n x_n - s_n$ and $w_{n,e} \leftarrow w_{n,e} + x_n - z_e$.

### What is exchanged

Per iteration, each node sends the vector $x_n + w_{n,e}$ to the neighbour at the other
end of link $e$: $p = 5$ numbers per direction per link in this project. Raw samples, labels
and margins never leave the node.

### What is reported

The nodes hold slightly different models until they converge. The plots evaluate the
global objective at the average $\bar{x} = \frac{1}{M}\sum_n x_n$ and report the
disagreement $\max_n \lVert x_n - \bar{x} \rVert$ separately. The average is used for
reporting only; the algorithm never computes it.
