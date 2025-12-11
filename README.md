# Deception Detection in Dyadic Exchanges Using Multimodal Machine Learning

This repository contains the code accompanying the paper:

**"Deception Detection in Dyadic Exchanges Using Multimodal Machine Learning: A Study on a Swedish Cohort"**  
Preprint available on [arXiv](https://arxiv.org/abs/2506.21429).

The goal of this repository is to increase the reproducibility of our experiments and provide a reference implementation for future research on multimodal deception detection, particularly in dyadic contexts.

> ⚠️ **Note**: The dataset used in this study is protected and cannot be shared due to ethical and privacy restrictions.

---

## 🧠 Overview

The paper explores multimodal machine learning techniques to detect deception in dyadic interactions, using a unique Swedish-language dataset. It compares early vs. late fusion strategies and examines the impact of incorporating features from both participants in a conversation.

The codebase includes:
- Feature extraction and preprocessing
- Model training and evaluation
- Comparison of fusion strategies
- Scripts for reproducibility and persistent use

---

## 📁 Repository Structure

- `notebooks/`: Jupyter notebooks to run experiments interactively and explore results
- `src/`: Python modules for feature extraction, data loading, training, and evaluation
- `requirements.txt`: All required packages for running the project

---

## ⚙️ Setup

### 1. Clone the repository

git clone https://github.com/your-username/deception-detection-mm.git
cd deception-detection-mm

### 2. Install dependencies
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

### 3. Usage
You can either:

🔬 Run the experiments interactively:
Use the Jupyter notebooks to run and visualize the experiments step-by-step.

🧪 Run persistent scripts:
Use the Python scripts to run full training/evaluation pipelines.

## 📖 Citation

If you use this code in your research, please cite the following paper:

> **Rugolon, F.**, et al. (2025).  
> _Deception Detection in Dyadic Exchanges Using Multimodal Machine Learning: A Study on a Swedish Cohort_.  
> [arXiv:2506.21429](https://arxiv.org/abs/2506.21429)

```bibtex
@misc{rugolon2025deception,
  title={Deception Detection in Dyadic Exchanges Using Multimodal Machine Learning: A Study on a Swedish Cohort},
  author={Rugolon, Franco and [co-authors]},
  year={2025},
  eprint={2506.21429},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```


### Acknowledgments

```markdown
This work was conducted as a collaboration between the  
__Department of Psychology__ and the __Department of Computer and Systems Sciences__, Stockholm University.

The project was made possible by funding from the **Marcus and Amalia Wallenberg Memorial Foundation** (grant **MAW 2022.0062**).
```
