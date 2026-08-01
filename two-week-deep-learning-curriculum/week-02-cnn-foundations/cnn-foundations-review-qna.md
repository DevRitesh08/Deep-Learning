# Week 2 Review Q&A

## 1. Why use a CNN instead of a Dense layer directly on an image?

**Short answer:** CNNs preserve local spatial structure and reuse the same filter across positions, greatly reducing parameters.

**Build it out:** A dense layer treats flattened pixels as unrelated positions. A convolution learns a local pattern and scans it over the image, giving feature maps that retain spatial axes.

**Trap:** CNNs are not only for RGB photos; they are useful for many grid-like signals.

## 2. What is the shape of a batch of 64 RGB images of size 32x32 in `channels_last` format?

**Short answer:** \((64,32,32,3)\).

**Build it out:** Axis order is batch, height, width, channels. The last 3 is red, green, and blue channels, not three images.

**Trap:** A grayscale image has one channel, so its per-image shape is \((32,32,1)\), not \((32,32)\) when a CNN expects a channel axis.

## 3. What does one 3x3x3 RGB filter produce?

**Short answer:** One 2D output feature map.

**Build it out:** The filter has weights for all three input channels. At each location, it multiplies a 3x3x3 patch, sums all values, adds a bias, and writes one scalar.

**Trap:** One RGB filter does not produce three output maps. The number of filters determines output channels.

## 4. Calculate the valid-convolution output size for 28x28 input, 5x5 kernel, stride 1.

**Short answer:** 24x24.

**Build it out:** \(28-5+1=24\). With `valid` padding there is no border padding.

**Trap:** This calculation is per spatial axis; feature-map count comes from number of filters.

## 5. What does `same` padding do?

**Short answer:** With stride 1, it pads so the output has the input’s spatial height and width.

**Build it out:** Zero padding lets kernel centers reach border locations. It controls output shape but adds artificial boundary values.

**Trap:** Padding does not make lost information reappear; it changes how borders are handled.

## 6. Compare max, average, and min pooling.

**Short answer:** Max keeps the largest value, average computes the mean, and min keeps the smallest value.

**Build it out:** They have no learned kernel weights and apply separately to each channel. Max pooling is common; min pooling is unusual.

**Trap:** Mean pooling is average pooling, not min pooling.

## 7. Does max pooling make a CNN translation invariant?

**Short answer:** It can provide limited local translation tolerance, not full invariance.

**Build it out:** A strong activation moved within the same pooling window may still be retained. A larger move can change the output substantially.

**Trap:** “CNNs recognize an object anywhere automatically” overstates what architecture alone guarantees.

## 8. What is a feature map?

**Short answer:** The spatial response produced by one learned filter over an input.

**Build it out:** For 32 filters, a convolution layer outputs 32 feature maps. Their values are activations, not fixed edge labels.

**Trap:** Feature maps are not model parameters; kernels and biases are parameters.

## 9. What does `Flatten()` do?

**Short answer:** It reshapes each sample’s non-batch axes into one vector.

**Build it out:** \((B,8,8,32)\) becomes \((B,2048)\), allowing a Dense layer to consume the features.

**Trap:** Flattening has no trainable parameters, but the following Dense layer can have many.

## 10. Which output/loss pair is appropriate for three exclusive classes with integer labels 0, 1, 2?

**Short answer:** A 3-unit softmax output and sparse categorical cross-entropy.

**Build it out:** The output has shape \((B,3)\); each row is a probability distribution after softmax. Integer labels are compatible with the sparse loss.

**Trap:** Use sigmoid with binary cross-entropy for independent multi-label targets, not mutually exclusive classes.

## 11. Which neural-network architecture belongs to this week, and where do the others fit?

**Short answer:** CNNs are this week's focus because images have spatial grid structure. FNNs suit fixed features; RNNs model ordered sequences; transformers use attention over tokens or patches; GANs pair a generator with a discriminator.

**Build it out:** These families encode different assumptions about input structure. Knowing their roles prevents treating every architecture as a generic "better neural network."

**Trap:** A model family name does not automatically decide the task; targets, data, loss, and evaluation also matter.

## 12. Contrast classification, detection, segmentation, and tracking.

**Short answer:** Classification outputs an image label; detection outputs object boxes/classes; segmentation labels pixels; tracking keeps object identities across video frames.

**Build it out:** All can use CNN feature extractors, but their supervision and output heads differ. A video-tracking system also needs a way to connect objects over time.

**Trap:** A bounding box is not a segmentation mask, and one image-level label does not locate an object.

## 13. When should you scale, min-max normalize, or standardize pixels?

**Short answer:** Use `x / 255.0` for known 8-bit image values, min-max normalization for a deliberately chosen observed range, and standardization for training-set mean and standard deviation. Follow pretrained-model preprocessing exactly.

**Build it out:** Fit any data-dependent statistics only on the training split, then reuse them for validation and test data.

**Trap:** Computing \(\mu\), \(\sigma\), minimum, or maximum across the full dataset leaks information from evaluation data.

## 14. Compare the parameter count of one 3x3 RGB filter with a Full HD flattened Dense layer.

**Short answer:** One 3x3 RGB filter has 27 weights plus one bias. Flattening a 1920x1080x3 image into 128 Dense units needs 796,262,528 parameters in that Dense layer.

**Build it out:** CNN parameter sharing reuses one local filter at many positions; a Dense layer assigns independent connections from every flattened input to every unit.

**Trap:** The 6,220,800 image values are inputs, not automatically learned features or trainable parameters.

## Whiteboard drill

Trace \((B,64,64,3)\) through:

```text
Conv2D(8, 3, same) -> MaxPool2D(2) -> Conv2D(16, 3, valid) -> MaxPool2D(2)
```

State every tensor shape, total number of output feature maps after each convolution, and which layers have trainable weights.
