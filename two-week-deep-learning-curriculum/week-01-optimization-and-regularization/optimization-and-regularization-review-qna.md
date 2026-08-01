# Week 1 Review Q&A

## 1. What is the difference between an epoch and an optimizer step?

**Short answer:** An epoch processes the training set once; a step updates parameters once. With mini-batches, an epoch has many steps.

**Build it out:** For \(N\) examples and batch size \(B\), there are \(\lceil N/B\rceil\) updates per epoch. Full-batch gradient descent is the special case with one update per epoch.

**Trap:** “Iteration always means epoch.” Frameworks usually use iteration/step for one batch update.

## 2. Why is mini-batch SGD the practical default?

**Short answer:** It balances memory use, vectorized hardware efficiency, and useful gradient noise.

**Build it out:** Full-batch gradients may be expensive; single-example SGD underuses accelerators and is noisy. A mini-batch estimates the dataset gradient without needing all examples in memory.

**Trap:** A larger batch is not automatically better. It changes memory, update frequency, and often the useful learning-rate range.

## 3. What does Momentum store?

**Short answer:** A moving average of gradient direction, often called velocity.

**Build it out:** \(v_t=\mu v_{t-1}+g_t\); the parameter follows \(-v_t\). Consistent directions accumulate, while alternating noise partly cancels.

**Trap:** Momentum does not use squared gradients. That is the adaptive-scale idea used by RMSProp/Adam.

## 4. Why can AdaGrad stop too early?

**Short answer:** Its per-parameter sum of squared gradients never decreases.

**Build it out:** The denominator \(\sqrt{r_t}+\epsilon\) grows over time, shrinking the effective step. This can help sparse features but can make late training extremely slow.

**Trap:** AdaGrad’s base learning rate is not literally removed; it is still a hyperparameter in the update.

## 5. How does RMSProp address AdaGrad’s weakness?

**Short answer:** It uses a decaying moving average of squared gradients instead of a lifetime sum.

**Build it out:** Recent gradient scale controls the denominator, so old large gradients gradually lose influence.

**Trap:** RMSProp is not “Momentum plus AdaGrad.” Adam is the familiar method that combines first- and second-moment estimates.

## 6. Why does Adam use bias correction?

**Short answer:** Its moving averages start at zero and are biased low in early updates.

**Build it out:** Dividing \(m_t\) by \(1-\beta_1^t\) and \(v_t\) by \(1-\beta_2^t\) corrects this initialization effect.

**Trap:** The correction is most important at the beginning, not a substitute for a good learning rate.

## 7. What causes exploding gradients?

**Short answer:** Repeated chain-rule products can amplify gradient magnitude across many layers or time steps.

**Build it out:** Large weights, unsuitable initialization, unstable learning rates, or certain architectures can make the local Jacobian products grow. Watch gradient norms and numerical errors.

**Trap:** Gradient clipping limits a symptom; it may not cure the underlying configuration error.

## 8. When do you choose Glorot versus He initialization?

**Short answer:** Start with Glorot for roughly symmetric activations and He for ReLU-family activations.

**Build it out:** Both aim to preserve useful variance, but ReLU removes part of the signal distribution, motivating He’s larger variance scale.

**Trap:** Initializing all weights to small identical values still fails because symmetry remains.

## 9. What does `Dropout(0.3)` mean in Keras?

**Short answer:** During training it randomly zeros 30% of the layer inputs and rescales the kept activations; at inference it does nothing.

**Build it out:** The rescaling is inverted dropout: kept values are divided by \(1-0.3\). This keeps the expected activation scale aligned with inference.

**Trap:** Do not manually multiply weights by 0.7 at inference when using Keras’s standard Dropout layer.

## 10. How do you decide whether dropout helped?

**Short answer:** Compare validation behavior under a controlled experiment.

**Build it out:** Keep data split, seed where practical, architecture, and training budget stable. If training performance falls slightly while validation performance improves, the regularization may help.

**Trap:** A smaller train-validation gap alone is not success if both metrics became worse.

## Rapid-fire drill

1. Write Adam’s two state updates from memory.
2. State one reason SGD is noisy.
3. State one cause and one remedy for exploding gradients.
4. Give the Keras meaning of `rate=0.5`.
5. Explain why the initialization choice is tied to activation choice.
