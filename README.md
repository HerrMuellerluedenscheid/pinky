Prereqs
-------

 - pyrocko
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
