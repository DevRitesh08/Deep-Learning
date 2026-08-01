# Week 2: CNN Foundations

## Outcome

By the end of this week, you can place CNNs among major neural-network families, distinguish computer-vision task types, represent grayscale/RGB images as tensors, preprocess pixels without leakage, calculate convolution and pooling output shapes, explain parameter sharing and local receptive fields, and trace a complete RGB classifier from input to softmax.

## Scope map

- **Owned:** supplied transcripts 31-39.
- **Prerequisite recap only:** Week 1 training loop, ReLU, softmax, cross-entropy, and backpropagation.
- **Orientation only:** FNNs, RNNs, transformers, GANs, and the difference among classification, detection, segmentation, and tracking.
- **Handoff beyond this week:** data augmentation, batch normalization, transfer learning, detection, segmentation, tracking, and modern CNN architectures.

## Seven-day plan

| Day | Focus | Evidence of understanding |
| --- | --- | --- |
| 1 | Architecture and CV orientation | Choose CNNs for spatial grids; distinguish classification, detection, segmentation, and tracking. |
| 2 | Image tensors and preprocessing | Write grayscale/RGB/batch shapes and choose scaling versus standardization without leakage. |
| 3 | Convolution | Hand-calculate a small valid cross-correlation output and name each output channel. |
| 4 | Padding and pooling | Calculate output shape with stride/padding and correct the min-vs-average pooling confusion. |
| 5 | Flattening and dense heads | Trace feature-map shape to Dense input size; explain when global pooling is an alternative. |
| 6 | RGB classifier | Trace the supplied-style RGB pipeline and choose loss/activation for three classes. |
| 7 | Integration review | Complete the shape drill and answer the final 90-second challenge. |

## Files

- [Technical notes](cnn-foundations-notes.md)
- [Interview and exam review Q&A](cnn-foundations-review-qna.md)
- [Visual resources](visual-resources.md)

## Completion check

- Derive the spatial output formula from input, kernel, padding, stride, and dilation.
- Explain the shape of a kernel for RGB input and why it produces one output feature map.
- Explain why max pooling gives limited local translation tolerance, not perfect global invariance.
- Distinguish 8-bit scaling, min-max normalization, and training-split standardization.
- Complete the shape trace in the notes without looking.
