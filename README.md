# Pneumonia Detection from Chest X-Ray Images Using CNN & Grad-CAM

**Deep Learning Implementation & Academic Project Report**  
**Author**: Ritam Bhoumik (Student Code: `BWU/BCA/24/160`)  
**Department**: Department of Computational Sciences, Brainware University, Kolkata  

---

## Repository Contents

| File | Format | Description |
| :--- | :---: | :--- |
| **[`Pneumonia Detection.ipynb`](./Pneumonia%20Detection.ipynb)** | Notebook | Complete Google Colab / Jupyter Notebook with data pipeline, ResNet-18 transfer learning model, 8-epoch training, independent test evaluation, ROC curve, confusion matrix, and Grad-CAM explainability heatmaps. |
| **[`PROJECT REPORT.md`](./PROJECT%20REPORT.md)** | Markdown | Full academic project report with all chapters, formatted tables, and embedded high-resolution figures. |
| **[`Pneumonia Detection Project Report.pdf`](./Pneumonia%20Detection%20Project%20Report.pdf)** | PDF | Printable academic project report document formatted for Brainware University. |
| **[`Pneumonia Detection Project Report.docx`](./Pneumonia%20Detection%20Project%20Report.docx)** | Word | Editable Microsoft Word document of the project report. |

---

## Model & Performance Summary

- **Architecture**: Pretrained ResNet-18 with custom classification head (512 to 128 to 1), Batch Normalization, ReLU, and Dropout (0.4).
- **Dataset**: Kaggle Chest X-Ray Pneumonia (Kermany et al.) resized to 224x224 with ImageNet normalization and clinical data augmentations (affine shifts, rotations, horizontal flips, contrast jitter).
- **Optimization**: AdamW (learning rate: 1e-4, weight decay: 1e-2), ReduceLROnPlateau scheduler, and class-weighted BCE loss (`pos_weight = 0.340`).
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) on the final residual block (`layer4[-1].conv2`) providing visual heatmaps of pulmonary infiltrates and consolidations.

### Independent Test Set Results (624 Images)

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Accuracy** | **88.14%** | 550 of 624 test images classified correctly |
| **Sensitivity (Recall)** | **99.23%** | 387 of 390 pneumonia cases correctly detected |
| **Specificity** | **69.66%** | 163 of 234 normal cases correctly identified |
| **F1-Score** | **0.9127** | Harmonic mean of precision and recall for Pneumonia |
| **ROC-AUC** | **0.9621** | Separability across all classification thresholds |

---

## How to Run the Notebook

Open [`Pneumonia Detection.ipynb`](./Pneumonia%20Detection.ipynb) directly in Google Colab or Jupyter Notebook:
1. Ensure GPU acceleration is enabled (`Runtime` -> `Change runtime type` -> `T4 GPU`).
2. Run cells sequentially to download the dataset via `kagglehub`, train the network, and generate Grad-CAM diagnostic visualizations.
