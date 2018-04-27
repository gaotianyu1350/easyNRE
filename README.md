# easyNRE

## Instructions

### Input Data

All input data should be placed directly in the same directory (specified in argument `export_data`). Different parts of the input data are put in different numpy data files:

* `vec.npy`: the word embedding matrix (shape: `(W, K)`)
* `train_instance_triple.npy`: unused (shape: `(T, X)`)
* `train_instance_scope.npy`: (shape: `(T, 2)`)
* `train_len.npy`: a vector containing the length of each training sentence (shape: `(N,)`)
* `train_label.npy`: a vector containing the label of each training (shape: `(N,)`)
* `train_word.npy`: a matrix containing the index of each word in each training sentence (shape: `(N, L)`)
* `train_pos1.npy, train_pos2.npy`: position of each word relative to entity 1/2 in each sentence, which would be used in position embeddings (shape: `(N, L)`)
* `train_mask.npy`: a matrix containing 0, 1, 2 only that indicates the position of each word in each training sentence (in PCNN: 0 for out of bound, 1 for head, 2 for body, 3 for tail) (shape: `(N, L)`)

Notes on the parameters above:

* `N`: the number of training sentences
* `L`: the max length of training sentences
* `K`: the dim of the word embedding space
* `W`: the number of words
* `R`: the number of different relations
* `T`: the number of relation triples (bags)
* `X`: whatever

### Configuration

A configuration file named `config` should be placed under the data directory.

* `fixlen`
* `maxlen`
* `relation2id`