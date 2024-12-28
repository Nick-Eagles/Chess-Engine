## TODO

- `Game.toNN_vecs(every = True)` unfortunately needs unit tests. It's far too
complex to assume it just works

## Misc thoughts/worries:

- Despite the policy network picking a master's move as the top move ~25% of
the time, Traversals appear to miss obvious captures a high percentage of the
time. Most likely, master games don't frequently involve positions where
material is hung, and yet an effective Traversal depends on knowing how to
grab material when it's available for free
- I've seen the engine select bad moves that appeared good simply because of
limited search depth (e.g. the final move/ edge in the search tree is a
rewarding capture which would then be recaptured with equal or greater reward
were the tree to cover that far). Setting "gamme_exec" high exacerbates this
problem, yet setting it low leads to reckless material grabbing. Note that this
problem goes completely away in the limit that the network's value output
becomes perfectly accurate, regardless of "gamma_exec". However, the problem is
highly noticeable when "gamma_exec"=1 even when value certainty ~= 0.5.

