# Transformer Math Evaluation
> Jonathan Drechsel, Anja Reusch, Steffen Herbold

[![arXiv](https://img.shields.io/badge/arXiv-2502.20855-B31B1B.svg)](https://arxiv.org/abs/2502.20855)


Framework to evaluate mathematical aware Transformer models, first introduced by [MAMUT: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training](https://arxiv.org/abs/2502.20855).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aieng-lab/transformer-math-evaluation
cd transformer-math-evaluation 
```

#### 2. Create a Conda Environment:
```bash
conda create --name math-eval python=3.10
conda activate math-eval
conda install pip
pip install -r requirements.txt
```

### 3. Data Generation
```bash
python src/util/generate_data.py
```
This script generates the data splits of the large pre-training datasets [NMF](https://huggingface.co/datasets/ddrg/named_math_formulas) and [MFR](https://huggingface.co/datasets/ddrg/math-formula-retrieval). However, for NMF, a special split is also available on [Hugging Face](https://huggingface.co/datasets/ddrg/named_math_formulas_ft) along with additional meta data (e.g., which [MAMUT](https://arxiv.org/abs/2502.20855) strategies have been used to generate a false example). This enriched data is used by default in the config files.

## Usage

Everything can be controlled by [executor.py](src/executor.py). 

```python src/executor.py -model bert-base-cased -config config/all.json -data_dir data```

- `-model`: The model to be evaluated.
- `-config`: The configuration file to be used.
- `-data_dir`: The directory containing the data to be evaluated.

The following base configurations are available:

- `config/nmf.json`: Named Math Formula (NMF) retrieval, i.e., an IR task with a name of a mathematical formula as query (e.g., "Binomial Formula") and the formula itself as document (e.g., $(\alpha + z)^2 = z^2 + \alpha^2 + 2\cdot \alpha \cdot z$).
- `config/nmf-split.json`: NMF with a special train/val/test split such that an identity is only in one of the splits.
- `config/nmf-fp1.json`: NMF using the same false examples for each epoch (the default NMF task changes false examples every epoch)
- `config/nmf-no-challenging`: NMF using non-challenging false examples, i.e., random formulas of different mathematical identities.
- `config/mfr.json`: Math Formula Retrieval (MFR), i.e., an IR task with formulas for both, query and document (e.g., query $n!=1\cdot \dots \cdot n$ and document $n!\coloneqq \prod_{k=1}^n k$).


Notice that there are more evaluations already available (not well documented) and more evaluation methods are planned for the future.
For example, Evaluates a model on the Mathematical Structure Attention Score, aiming to find mathematical aware attention heads. Notice that this evaluations are not published as part of the MAMUT paper.


## CITATION
If you use this evaluation framework, please cite the following paper:
```bibtex
@misc{drechsel2025mamutnovelframeworkmodifying,
      title={{MAMUT}: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training}, 
      author={Jonathan Drechsel and Anja Reusch and Steffen Herbold},
      year={2025},
      eprint={2502.20855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20855}, 
}
```
