# Deep Learning: Two-Week Curriculum

This curriculum turns the supplied optimizer, training-stability, regularization, and CNN lessons into two complete study weeks. It assumes that you already understand forward propagation, backpropagation, loss functions, activations, and basic ANN classification/regression.

## Boundaries

- **Week 1 owns:** gradient-descent variants, optimizer state, exploding gradients, initialization, and dropout.
- **Week 2 owns:** image tensors, convolution, padding, pooling, CNN-vs-ANN reasoning, flattening/dense heads, and an RGB CNN trace.
- **Not changed:** `ANN Regression and classification` remains intact and is not a dependency to edit.
- **Not covered yet:** CNN architectures, data augmentation, transfer learning, object detection, and segmentation.

## Start here

| Unit | Main outcome | Core files |
| --- | --- | --- |
| [Week 1 - Optimization and Regularization](week-01-optimization-and-regularization/README.md) | Choose, explain, and debug an optimizer, initialization, and dropout configuration. | [Notes](week-01-optimization-and-regularization/optimization-and-regularization-notes.md) · [Review Q&A](week-01-optimization-and-regularization/optimization-and-regularization-review-qna.md) |
| [Week 2 - CNN Foundations](week-02-cnn-foundations/README.md) | Trace an image tensor through a CNN and calculate its output shapes. | [Notes](week-02-cnn-foundations/cnn-foundations-notes.md) · [Review Q&A](week-02-cnn-foundations/cnn-foundations-review-qna.md) |

Read [Sources](sources.md) alongside the notes. The temporary copies of your PDFs and screenshots are in `tmp/source-assets/`; each unit keeps its original teaching diagram in its own `tmp/` folder. They are deliberately kept separate from the learner-facing Markdown.

## Transcript coverage

| Supplied lessons | Curriculum home |
| --- | --- |
| 21-27: Gradient descent through Adam | Week 1, Days 1-4 |
| 28-30: Exploding gradients, initialization, dropout | Week 1, Days 5-6 |
| 31-39: CNN introduction through RGB example | Week 2, Days 1-6 |

## Study loop

1. Read the day section in the unit README.
2. Reconstruct the key formula or tensor shape without looking.
3. Complete the day’s prompt in the notes.
4. Answer the matching review questions aloud.
5. Use the final-day drill to decide whether to revisit the week.

The curriculum progress is recorded in `.curriculum-progress.json`.
