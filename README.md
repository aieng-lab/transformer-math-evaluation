# transformer-math-evaluation

Repository to evaluate mathematical aware Transformer models. 

Everything can be controlled by [executor.py](src/executor.py). 

```python executor.py --model bert-base-cased --config config/default.json --data_dir data```

The most important config ids are 
- formula-ir
- math-structure-score

# formula-ir
Fine-tunes models on a mathematical ir task. Supports options like key_lhs and key_rhs for keys in the data used for query and documents, respectively.

# math-structure-score
Evaluates a model on the Mathematical Structure Attention Score. You can also directly run [structure.py](src/analysis/structure.py)
