
# MODEL ARCHITECTURE

## Final Recommended Architecture

Image Encoder:
- EfficientNet-B0

Text Encoder:
- PhoBERT

Fusion:
- Feature Concatenation

Classifier:
- Fully Connected Layer
- Softmax

---

# Why This Architecture?

- Good balance between accuracy and compute
- Suitable for Vietnamese NLP
- Easier training than BLIP/large VLMs
- Strong academic novelty

---

# Why NOT Video Models?

Not recommended:
- C3D
- R3D
- R(2+1)D
- MoviNet
- MambaVideo

Reason:
Dataset uses static rice leaf images, not video sequences.

