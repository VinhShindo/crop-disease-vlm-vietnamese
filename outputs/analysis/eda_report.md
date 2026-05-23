# MULTIMODAL EDA REPORT

# 1. Dataset Overview

## Dataset Name
Rice Leaf Diseases Dataset (Multimodal Extension)

## Total Classes
4

## Total Images
3355

## Disease Classes

- BrownSpot
- Healthy
- Hispa
- LeafBlast

---

# 2. Dataset Statistics

| Class | Number of Images | Percentage |
|---|---:|---:|
| BrownSpot | 523 | 15.59% |
| Healthy | 1488 | 44.35% |
| Hispa | 565 | 16.84% |
| LeafBlast | 779 | 23.22% |

## Observations

- Healthy class contains the largest number of samples.
- BrownSpot and Hispa classes contain fewer samples.
- Dataset imbalance exists and may affect classification performance.
- Stratified data split is required during training.

---

# 3. Image Analysis

## Average Resolution

- Width: 2049
- Height: 2049

## Image Quality

- No corrupted image detected.
- Images are visually clean.
- Leaf regions are clearly visible.
- Dataset quality is suitable for deep learning tasks.

## Visual Characteristics

The dataset contains:
- varying leaf orientations
- different disease severity levels
- multiple texture patterns
- different leaf colors
- symptom variations across classes

## Observations

### Advantages

- High image resolution
- Clear disease patterns
- Low image corruption
- Suitable for CNN and Transformer architectures

### Limitations

- Background conditions are relatively simple
- Limited environmental diversity
- Few real-world field conditions
- Limited lighting variations

---

# 4. Text Metadata Analysis

## Metadata Summary

Total Metadata Samples:
3355

Average Texts Per Image:
4.0

Average Text Length:
10.73 words

Language:
Vietnamese

## Metadata Structure

Each image contains:
- image path
- disease label
- multiple Vietnamese descriptions
- multimodal metadata

Example:

```json
{
  "image": "BrownSpot/img_001.jpg",
  "texts": [
    "Lá lúa xuất hiện nhiều đốm nâu nhỏ.",
    "Phiến lá có các vùng cháy màu nâu.",
    "Triệu chứng bệnh đốm nâu xuất hiện rõ."
  ],
  "label": "BrownSpot"
}
````

---

# 5. Text Dataset Observations

## Advantages

* Multiple descriptions per image improve semantic diversity.
* Vietnamese symptom descriptions support NLP learning.
* Metadata increases multimodal research novelty.
* Text information can improve disease discrimination.

## Semantic Information Included

Descriptions contain:

* disease symptoms
* lesion appearance
* color changes
* texture abnormalities
* infection severity
* visual characteristics

## Importance for Vision-Language Learning

The text metadata enables:

* image-text alignment
* multimodal embedding learning
* semantic disease understanding
* Vietnamese agricultural NLP research

---

# 6. Dataset Challenges

## 1. Class Imbalance

Healthy class contains significantly more samples.

Potential impacts:

* model bias toward Healthy class
* reduced recall for minority disease classes
* unstable macro metrics

Recommended solutions:

* weighted loss
* focal loss
* oversampling
* data augmentation
* class-balanced training

---

## 2. Domain Gap

Current dataset characteristics:

* mostly clean backgrounds
* controlled capture conditions
* limited real-field complexity

Potential problems:

* reduced generalization in real farms
* sensitivity to lighting/environment changes

Recommended solutions:

* field data collection
* domain adaptation
* strong augmentation
* real-world validation

---

## 3. Synthetic Metadata Limitation

Current Vietnamese descriptions are generated programmatically.

Potential limitations:

* repetitive linguistic patterns
* limited contextual diversity
* lack of farmer-style descriptions

Recommended improvements:

* collect real agricultural reports
* add weather information
* add environmental context
* include farmer observations

---

# 7. Recommended Training Strategy

## Image Encoder

Recommended:

* EfficientNet-B0

Alternative:

* Vision Transformer (ViT)
* ConvNeXt
* Swin Transformer

---

## Text Encoder

Recommended:

* PhoBERT

Alternative:

* viBERT
* XLM-R
* multilingual BERT

---

## Fusion Strategy

Recommended:

* Cross Attention
* Feature Concatenation

Advanced options:

* Multimodal Transformer
* BLIP-style Fusion
* CLIP-style Alignment

---

## Recommended Input Resolution

* 224x224
* 256x256

---

## Recommended Batch Size

Depending on GPU:

* 16
* 32
* 64

---

# 8. Recommended Evaluation Metrics

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score

## Main Metric

F1-score

Reason:
Dataset imbalance makes F1-score more reliable than accuracy.

## Additional Evaluation

* Confusion Matrix
* Per-class Recall
* Macro F1-score
* Weighted F1-score

---

# 9. Expected Research Contributions

This project contributes to:

* Vietnamese agricultural AI
* Vision-Language learning
* Multimodal disease classification
* Smart farming applications
* Vietnamese NLP for agriculture

## Research Novelty

The main novelty includes:

* multimodal rice disease recognition
* Vietnamese symptom descriptions
* image-text fusion for agriculture
* domain-specific Vietnamese metadata generation

---

# 10. Generated Files

## Visualizations

* class_distribution.png
* dataset_overview.png
* image_resolution_distribution.png
* text_length_distribution.png
* texts_per_image_distribution.png

## Analysis Files

* dataset_statistics.csv
* text_statistics.csv
* corrupted_images.csv
* eda_report.md

---

# 11. Conclusion

The dataset is suitable for:

* rice leaf disease classification
* multimodal learning
* Vietnamese Vision-Language research
* agricultural AI systems
* deep learning experimentation

The dataset quality, image resolution, and Vietnamese metadata
provide a strong foundation for developing a Vision-Language
model for smart agriculture applications.