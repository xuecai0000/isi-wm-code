# Background data protocol

## Video pool

100 natural video clips, 352x352. Schema-2 split (frozen JSON manifests):

| Split | Videos |
|---|---|
| train (B_train) | video0-79 |
| validation | video80-84 |
| support | video85-89 |
| test (unseen) | video90-99 |

## Compositor

Blue-key mask on the clean DMC render: a pixel is background if B>G and B>R.
The current video frame replaces the masked region.  Backgrounds PLAY during
the episode: one source frame is advanced per control step from a random
valid start; clips loop.

## Intervention pairing contract

Each intervention root provides:
- Matched clean/video renders of the same physical state (background twin)
- Genuinely executed one-step-divergent action branches (+m*e_i / -m*e_i)
  followed by shared continuation actions, for every action axis
- Simulator-measured normalized outcome gap g between every ordered branch pair
- Binary eligibility q = 1[g > 0.05]

The training capsule stores only RGB stacks, actions, branch identities, g,
and q.  Raw simulator states are consumed at construction time and never
enter the model.

## Evaluation conditions

| Condition | Background | Episodes |
|---|---|---|
| clean | none | 20 |
| seen | train pool, fresh episodes | 20 |
| unseen | test pool (video90-99) | 20 |

Cup Catch uses 50 episodes per condition (bimodal returns).
