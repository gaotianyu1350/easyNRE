# easyNRE

An open-source framework for neural relation extraction.

## Overview

It is a TensorFlow-based framwork for easily building relation extraction models. We divide the pipeline of relation extraction into four parts, which are embedding, encoder, selector and classifier. For each part we have implemented several methods.

* Embedding
  * Word embedding
  * Position embedding
  * Concatenation method
* Encoder
  * PCNN
  * CNN
  * RNN
  * BiRNN
* Selector
  * Attention
  * Maximum
  * Average
* Classifier
  * Softmax loss function
  * Output
  
All those methods could be combined freely. 

We also provide fast training and testing codes. You could change hyper-parameters or appoint model architectures by using Python arguments. A plotting method is also in the package.

## Installation

1. Install TensorFlow
2. Clone the easyNRE repository:
  ```
  git clone git@github.com:gaotianyu1350/easyNRE.git
  ```
3. Download NYT dataset from `https://drive.google.com/file/d/1BnyXMJ71jM0kxyJUlXa5MHf-vNjE57i-/view?usp=sharing`
4. Extract dataset to `./origin-data`
```
tar xvf origin_data.tar
```

## Quick Start

### Process Data

```
python gen_data.py
```
The processed data will be stored in `./data`

### Train Model
```
python train.py --model_name pcnn_att
```

The arg `model_name` appoints model architecture, and `pcnn_att` is the name of one of our models. All available models are in `./model`. About other arguments please refer to `./train.py`. Once you start training, all checkpoints are stored in `./checkpoint`.

### Test Model
```
python test.py --model_name pcnn_att
```

Same usage as training. When finishing testing, the best checkpoint's corresponding pr-curve data will be stored in `./test_result`.

### Plot
```
python draw_plot.py pcnn_att
```

The plot will be saved as `./test_result/pr_curve.png`. You could appoint several models in the arguments, like `python draw_plot.py pcnn_att pcnn_max pcnn_ave`, as long as there are these models' results in `./test_result`.

