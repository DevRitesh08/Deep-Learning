# Week 1: Optimization and Regularization

## Outcome

By the end of this week, you can explain how a parameter update is formed, distinguish batch/SGD/mini-batch training, derive Momentum/RMSProp/Adam state updates, diagnose unstable gradients, select an initialization, and use dropout correctly in training versus inference.

## Scope map

- **Owned:** supplied transcripts 21-30.
- **Prerequisite recap only:** loss, derivatives, backpropagation, activations, and ANN structure.
- **Handoff to Week 2:** CNNs use the same loss, optimizer, initialization, regularization, and backpropagation ideas.

## Seven-day plan

| Day | Focus | Evidence of understanding |
| --- | --- | --- |
| 1 | Batch gradient descent and SGD | Explain why an epoch and an optimizer update are not the same thing. |
| 2 | Mini-batches and Momentum | Compute steps per epoch and describe how momentum reduces zig-zagging. |
| 3 | AdaGrad and RMSProp | Explain why AdaGrad’s effective steps can become too small. |
| 4 | Adam | Reconstruct both moment estimates and the bias-correction reason. |
| 5 | Exploding gradients and initialization | Select Glorot or He from the activation and state one stabilizing action. |
| 6 | Dropout | State what `rate=0.3` means in Keras and what happens at inference. |
| 7 | Integration review | Diagnose the four cases in the notes and answer the final drill aloud. |

## Files

- [Technical notes](optimization-and-regularization-notes.md)
- [Interview and exam review Q&A](optimization-and-regularization-review-qna.md)
- [Visual resources](visual-resources.md)

## Practical handoff

Use the existing optimizer practice at `../../Complete Deep Learning/Practicals/Optimizers.ipynb` as a companion. Do not edit it for this curriculum. A productive experiment is to keep the model and data fixed while changing only the optimizer and learning-rate schedule.

## Completion check

- Derive the Adam update without notes.
- Explain why all-zero weights fail even when gradients exist elsewhere.
- Explain why Keras dropout has no inference-time scaling step.
- Answer the 90-second challenge in [Review Q&A](optimization-and-regularization-review-qna.md).
