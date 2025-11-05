# 🦷 From PDF to Dental View Classification: A Human-in-the-Loop Dataset and Pipeline for Oral Health Imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Made with ❤️ by NOVA IMS](https://img.shields.io/badge/Made%20with-%F0%9F%92%96%20by%20NOVA%20IMS-orange)](https://novaims.unl.pt)
[![Dataset](https://img.shields.io/badge/Dataset-SB%20Brasil%202023-green)]()

---

## 🧭 Overview

This repository contains the full source code, metadata, and documentation for the paper:

> **Souza, P. V. C. et al.**  
> *From PDF to Dental View Classification: A Human-in-the-Loop Dataset and Pipeline for Oral Health Imaging*  
> Submitted to the *IEEE Journal of Translational Engineering in Health and Medicine (JTEHM)*, 2025.

The project introduces a **reproducible, interpretable, and open pipeline** to transform public oral-health training manuals (from the *SB Brasil 2023* program) into a structured, machine-readable dataset suitable for computer vision research in dentistry and public health.

---

## 📁 Repository Structure

```
Pipeline-for-Oral-Health-Images/
│
├── codeextraction/           # Scripts for extracting and parsing images from PDF manuals
├── metadata/                 # Manifest files and metadata logs (volunteer, module, sequence)
├── images_by_view/           # Extracted intraoral photos organized by anatomical view
├── clusters/                 # Unsupervised clustering outputs and feature embeddings
├── unsupervised_kit/         # Tools for clustering, PCA, and feature-space exploration
├── review/                   # Human-in-the-loop labeling and dashboard tools
│
├── Pranchetas fotografias - dentição decídua - CPOD treinamento/
├── Pranchetas fotografias - PUFA - treinamento/
├── Pranchetas fotografias - traumatismo - treinamento/
│
├── features.npy              # Precomputed feature matrix (embeddings)
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation and Setup

### 1. Create a new environment

```bash
conda create -n oralhealth python=3.10
conda activate oralhealth
pip install -r requirements.txt
```

### 2. Extract and catalog images

```bash
python codeextraction/extract_and_catalog.py
```

### 3. Export samples for labeling

```bash
python export_for_labeling.py
```

### 4. Apply human labels and retrain model

```bash
python apply_labels_and_retrain.py
```

### 5. Launch the interactive dashboard

```bash
python dashboard.py
```

---

## 🧠 Features

✅ Automated extraction of intraoral images from public calibration manuals  
✅ Metadata preservation (module, volunteer, sequence)  
✅ Feature embedding and unsupervised clustering  
✅ Interactive human-in-the-loop labeling interface  
✅ Dashboard for visual analytics and dataset quality control  
✅ Baseline CNN model for dental view classification (frontal vs. occlusal)

---

## 📊 Example Outputs

- **Feature embeddings:** stored in `features.npy`  
- **Clustering visualization:** PCA and t-SNE projections for unsupervised inspection  
- **Dashboard:** view-level analytics and human labeling history  
- **Baseline performance:** 96.4% accuracy (frontal vs. occlusal classification)

---

## 📘 Citation

If you use this repository, please cite:

```bibtex
@article{souza2025oralhealth,
  author    = {Souza, Paulo Vitor de Campos and others},
  title     = {From PDF to Dental View Classification: A Human-in-the-Loop Dataset and Pipeline for Oral Health Imaging},
  journal   = {IEEE Journal of Translational Engineering in Health and Medicine},
  year      = {2025},
  note      = {Under Review}
}
```

---

## 🙏 Acknowledgment

This work was supported by national funds through the **Fundação para a Ciência e a Tecnologia (FCT)**, under project **UIDB/04152 – Centro de Investigação em Gestão de Informação (MagIC)**, NOVA Information Management School (NOVA IMS), Universidade Nova de Lisboa, Portugal.  

We also acknowledge the **Brazilian Ministry of Health** for providing access to calibration materials and public documentation from the *SB Brasil 2023* National Oral Health Survey, which made this research possible.

---

## ⚖️ License

This repository is released under the **MIT License**.  
You are free to use, modify, and distribute this code with proper citation of the original work.

---

## 📬 Contact

**Paulo Vitor de Campos Souza**  
NOVA Information Management School (NOVA IMS)  
Email: [psouza@novaims.unl.pt](mailto:psouza@novaims.unl.pt)

---

> ✳️ *“Bridging public health and computer vision for interpretable oral-health AI.”*
