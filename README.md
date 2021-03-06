# Graph Traversal with Tensor Functionals (GTTF)

GTTF is a framework for developing and using scalable graph learning models.

This is the public version. Stay tuned for coming updates!

## Usage

GTTF offers many popular off the shelf models (with more being added). These can be easily used for scalable training on graph structured data. 

### Simple Scalability for New Models:

To add scalability to an a new graph model, the simplest way is to use the `SampledAdjacency` accumulation class (e.g. `train_tf_gcn.py` lines 55-63). This uses GTTF to accumulate a sampled adjacency, then trims and untrims the adjacency and input features before and after running the model for additional speed-up. 

### Integrated Scalability:

In more advanced cases, there may be desire to integrate more fully with GTTF, and use its algorithmic versatility to develop novel algorithms or to scale more complicated graph learning models.

For further instruction, see `framework/README.md`.

## Experiments

The `experiments` directory contains shell scripts for running previously conducted graph learning experiments, including the ones from the original paper. They are labelled as `<model>_on_<dataset>.sh`. 

## Scalability Results

TODO: add final scalability table

## Reference

If GTTF contributed to the development of your research, please cite our paper

```
@inproceedings{
    GTTF2021,
    title={Graph Traversal with Tensor Functionals: A Meta-Algorithm for Scalable Learning},
    author={Elan Sopher Markowitz and Keshav Balasubramanian and Mehrnoosh Mirtaheri and Sami Abu-El-Haija and Bryan Perozzi and Greg Ver Steeg and Aram Galstyan},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=6DOZ8XNNfGN}
}
```


