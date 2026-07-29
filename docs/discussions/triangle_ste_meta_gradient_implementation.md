# Triangle STE Meta-gradient Implementation

This note is the implementation-oriented version of
`meta_learn_load_balance_ste_width.tex`.

## Goal

We want to adapt the load-balance STE width `h` online without adding an
extra forward pass. The intended schedule is:

1. At iteration `t`, run the normal forward/backward/update.
2. Save the smallest object needed to learn whether the iteration-`t` value of
   `h` helped.
3. At iteration `t + 1`, after the forward/backward gives the next LM gradient,
   use the saved object to update `h_t`.
4. Throw away the old object and save the new one for iteration `t + 1`.

## High Level Implementation

- We do not need to save the old computational graph across iterations.
- At iteration `t`, compute the sensitivity vector
  `s_t = d theta_{t+1} / d h_t`, detach it, and store it.
- At iteration `t + 1`, use the new LM gradient to compute
  `meta_grad_h_t = dot(grad_theta L_LM,t+1, s_t)`.
- Then update `h`, discard `s_t`, and save the new sensitivity vector for the
  next step.
- This same pattern also works for meta-learning the LB coefficient `alpha`;
  just save `s_alpha,t = d theta_{t+1} / d alpha_t`.
- The sensitivity-vector interface is the same whether we use an SGD
  approximation or an exact optimizer meta-gradient. The difference is only how
  we compute `s_t`.

## Option 1: SGD Approximation

- Approximate the optimizer as plain SGD for the meta-gradient calculation.
- For the STE width:
  `s_h,t = -lr * d/dh_t [grad_theta L_DLB,t(theta_t, h_t)]`.
- For the LB coefficient, if
  `grad_theta L_DLB = alpha * g0`, then
  `s_alpha,t = -lr * g0`.
- This is cheap and avoids differentiating through Muon/Adam/gradient clipping.
- It is approximate when the actual optimizer is not SGD, but it may still give
  a useful control signal, especially for `alpha`.

## Option 2: Exact Optimizer Meta-gradient

- Compute the exact sensitivity of the actual optimizer step:
  `s_t = d OptimizerStep(theta_t, grad_t(h_t), state_t) / d h_t`.
- This means `s_t` is the sensitivity of the post-optimizer parameters, not
  just the sensitivity of the raw gradient.

Two ways to do this:

- Functional optimizer: implement the optimizer step as a differentiable,
  side-effect-free function of parameters, gradients, and optimizer state, then
  differentiate through it.
- Manual exact calculation: derive and implement the optimizer step's local
  Jacobian-vector product, including the relevant optimizer state and update
  rules.

For Muon, both exact routes are non-trivial because the update is not a simple
elementwise transform of the gradient.

## Improvements

- Update `h` and `alpha` in log-space so they stay positive:
  optimize `log_h` and `log_alpha`, then use `h = exp(log_h)` and
  `alpha = exp(log_alpha)`.
- Use Adam for the scalar meta-parameters. This gives adaptive scaling and
  moment smoothing of the noisy per-step meta-gradients, so a separate EMA is
  probably not needed for the actual update.
- Update every `N` training steps instead of every step if the meta-gradient is
  too noisy. Accumulate or average the scalar meta-gradients across the window,
  then take one Adam step.
- Clamp the log-parameters to a reasonable range after each update, for example
  to keep `h` and `alpha` from collapsing to zero or exploding.
- Log the raw meta-gradient, Adam-smoothed update, current `h`, current
  `alpha`, and the dot product components. This makes it easier to tell whether
  the meta-signal is stable before trusting it for long runs.
