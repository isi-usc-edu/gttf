# Framework

This document describes how to develop models using the core GTTF framework. Each file also contains a description of its usage.

See our [paper](https://openreview.net/pdf?id=6DOZ8XNNfGN) for more information.

NOTE: There are some naming variations between the paper and the code. This document will use naming conventions from the codebase but note discrepencies.

## Walk Forest

As described in the paper, GTTF operates by building up a *Walk Forest* datastructure. 

The walk forest is represented as a list of tensors where each tensor has shape (batch_size, f * previous_width) and represents the results of the stochastic traversal from each node in the batch.

## Samplers

Samplers are functions that dictate how the stochastic walk is performed. This is almost the same as the notion of bias functions from the paper, but one level higher.

The Sampler interface must accept a compact adjacency, the current walk forest, and an integer fanout. It returns the next batch of nodes for the walk forest. If the previous walk forest depth had shape (B, D), the returned array should have shape (B, D*f).

We provide a uniform neighbor sampler as well as an interface for implementing samplers based on bias functions as described in the paper. 

While the paper describes bias functions as returning a transition probability distribution for each previous node, there is more flexibility here. For example, a Sampler could return a joint probability distribution across the whole batch and sample from that.

## Accumulation

Accumulation functions act on the walk forest to determine some result. The most frequent use cases are to build a sampled graph adjacency matrix or to calculate a loss. For extra performance the accumulation actually occurs on the full walk forest itself instead of being calculated with each step of the traversal.