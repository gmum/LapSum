# Experiments from the Paper


## Top-K Experiments (Tables 1-3)

For the Top-K experiments, you need to compile the CPP/CUDA implementation with the '*topk*' option enabled (line 60) in the [../lapsum/setup.py](../lapsum/setup.py#L60) configuration file. We have replaced the original top-k implementation with our enhanced version, which offers better performance and additional features.

The reference implementations and theoretical foundations can be found in:
- [DiffTopK Code](https://github.com/Felix-Petersen/difftopk)
- [Differentiable Top-k with Optimal Transport](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html) (See Supplemental)

---

## Soft-Permutation Experiments (Table 4)

### Setup Instructions
1. **Build CPP/CUDA Implementation**:
   - Enable the '*permute*' option (line 60) in [../lapsum/setup.py](../lapsum/setup.py#L60)
   - Requires CUDA toolkit and compatible GPU

2. **Configuration**:
   - Set paths in [./soft-permutation/run_permute.sh](./soft-permutation/run_permute.sh#L6-7):
     ```bash
     dataset="/path/to/your/dataset"  # Should contain train/val/test splits
     dir_results="/path/to/results"   # Will auto-create timestamped subfolders
     ```

3. **Execution**:
   ```bash
   bash ./soft-permutation/run_permute.sh
   ``` 

4. **Implementation Notes**
   - Based on [NeuralSort](https://github.com/ermongroup/neuralsort) wth upgrade to full PyTorch 2.5+ compatibility.
   - Original paper: '[Stochastic optimization of sorting networks
via continuous relaxations](https://arxiv.org/pdf/1903.08850)' (Figure 4)
   - Our improvements: [Our Paper](https://arxiv.org/abs/2503.06242) (Section 5.2)
