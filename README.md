# LapSum: One Method to Differentiate Them All - ICML 2025

This repository contains the official implementation of **LapSum**, a novel approach for differentiable ranking, sorting, and top-k selection, as presented in our ICML 2025 paper:

**"LapSum - One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection"**  
[arXiv preprint](https://arxiv.org/abs/2503.06242) | [ICML 2025 proceedings](https://proceedings.icml.cc/)  

## Repository Structure
```
.
├── experiments_of_paper/  # Reproduction scripts for paper experiments
│ └── README.md  # Instructions
├── lapsum/  # Core library implementation
│ └── README.md  # Installation and usage instructions
├── tutorial_notebooks/  # Pure PyTorch implementations with usage examples
└── README.md  # are you here
```

## Installation CPP/CUDA

The core implementation is available as the `lapsum` Python package. For detailed installation instructions, see [./lapsum/README.md](./lapsum/README.md)

**Tested Environment**:  
- OS: Ubuntu 24.04 LTS
- Python: 3.10+
- PyTorch: 2.3+
- CUDA: 12.3 (when using GPU acceleration)

## Tutorial Notebooks
For practical examples and educational purposes, see our Jupyter [./tutorial_notebooks/](./tutorial_notebooks/)

**🚀 Pro Tip:**  If you don't want to compile the CPP/CUDA implementation, you can use the **pure PyTorch version** of the library - it's **almost as fast and memory-efficient** as the CUDA implementation! 

For users less experienced with building libraries from source, we especially recommend using the this implementation - it works out-of-the-box with standard PyTorch installations and doesn't require any additional compilation steps!

## Paper Experiments
All experimental results from the paper can be reproduced using scripts in [./experiments_of_paper/](./experiments_of_paper/)

## License

This project is open-source under the **MIT License**. For full details, see the [LICENSE](./LICENSE) file.


### Citation

If you use LapSum in your research, please cite our work:

```bibtex
@inproceedings{lapsum2025,
  title={LapSum - One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection},
  author={\L{}ukasz Struski and Micha\l{} B. Bednarczyk and Igor T. Podolak and Jacek Tabor},
  booktitle={The International Conference on Machine Learning (ICML) 2025},
  year={2025}
}
