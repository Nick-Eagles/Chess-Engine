## TODO

- Find an external dataset of expert games and record in code how to retrieve
and parse it into input and output accepted by tensorflow/ `policy_net`
- Use a combined policy and value network moving forward (for simplicity and
code readability, don't even allow value networks)
- Don't use a resnet with residual blocks. Try the simplest architecture
possible before thinking of more complex improvements
- Remove scripts that aren't used and can't be identified in a reasonable
amount of time to be up to date and useful leads for the future
- Use very simple synthetic outputs (with the external games as inputs) to
verify a basic `policy_net` is even constructed correctly to learn basic
patterns. For example, to choose the ground-truth outputs for a given input,
use the following rule: if the current color has a queen, the policy should be
moving it to A1, and the evaluation should be that the current color wins.
Otherwise, move the current king to B2 and the opposite color wins. Note here
that these moves aren't even (and need not be) legal in general.
- `Game.toNN_vecs(every = True)` unfortunately needs unit tests. It's far too
complex to assume it just works. Use `every = False` in the above suggested
test 

## Misc thoughts/worries:

- while the architecture related to the policy output is efficient in terms of
using a low number of learnable parameters, information about the true
probability distribution over legal moves is lossily compressed in a way which
may be problematic, especially in particular positions. In reality, certain
start sqaures are associated only with particular end squares and particular
end pieces. However, these associations are totally lost (they are computed
independently) with the current architecture
- I've seen the engine select bad moves that appeared good simply because of
limited search depth (e.g. the final move/ edge in the search tree is a
rewarding capture which would then be recaptured with equal or greater reward
were the tree to cover that far). Setting "gamme_exec" high exacerbates this
problem, yet setting it low leads to reckless material grabbing. Note that this
problem goes completely away in the limit that the network's value output
becomes perfectly accurate, regardless of "gamma_exec". However, the problem is
highly noticeable when "gamma_exec"=1 even when value certainty ~= 0.5.

