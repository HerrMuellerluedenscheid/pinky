Prereqs
-------

 - tensorflow
 - scikit-optimize (optional for hyperparameter optimization)

Invoke
------

To start training:

    pinky --config <config_filename> --train

You can dump your examples to TFRecordDatasets to accelerate io operations:

    pinky --config <config_filename> --write <new_config_filename>

and use the newly created config file to run `--train`

Invoke pinky with `--debug` to enable keep track of weight matrices in
(tensorboard[https://www.tensorflow.org/guide/summaries_and_tensorboard].

Tests
-----

 - basic learning (synthetics, real data)
 - synthetics, layered cake, train with top and bottom layer containing events.
        Validate with events within middle layer
 - evaluation using 'unknown' velocity model
 - test extrapolation with fault plane geometry
 - synthetic pretraining
 - hyperparameter optimization using skopt

Outlook
-------

 - increase z error weight -> improve z estimates


Notes
-----

 - `batch_norm` is super important. Without, learning will get stuck at approx.
   500m x-error, y-error and z-error (in each domain!)
 - After waveform filtering (2-30Hz) results become worse. Why?
 - conv2D init set to None (Uses glorot (?)) smaller errors then
     truncated_normal(std=0.1)
 - strides 2 vs 1 on CNN: minor improvement in mean error. Max errors get little
   better. But it's much more expensive.
 - label normalization improves learning
 - large batch sizes (>64) perform worse than smaller  (<32) see:
   https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
