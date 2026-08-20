# Production readiness targets

Only integrate this recognizer with a system you own or are explicitly authorized to test.

## Current evidence

- The Beta + Default structural fallback reaches 97.37% exact accuracy on the
  38-image validation folder.
- It reaches 94.93% exact accuracy on all 138 currently labeled images.
- These images have already been inspected while tuning the pipeline, so neither
  number is an unbiased final production estimate.

## Recommended release gates

1. Keep at least 500 newly collected, correctly labeled images completely untouched
   until final evaluation. Prefer 1,000 or more, collected across multiple days and
   all visual styles.
2. For an assisted pilot with retry or manual review, require at least 98% exact-image
   accuracy on that untouched set.
3. For unattended one-shot use, target at least 99.5% exact-image accuracy. Character
   accuracy alone is not a sufficient release metric.
4. Require at least 99.5% correct output length and route every structurally invalid
   result to retry or review.
5. Measure accuracy separately for every visual style and every character. A high
   overall score must not hide a weak style or character.
6. On the deployment computer, target a warm P95 latency below 100 ms per image and
   test a long batch without crashes or steadily increasing memory use.
7. Continue sampling real results after release to detect style drift. Never train on
   the final test set.

The current system is suitable for local experiments and an authorized assisted
pilot. It is not yet ready for unattended production use.
