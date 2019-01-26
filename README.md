Prereqs
-------

 - [pyrocko](https://pyrocko.org/)
 - tensorflow
 - scikit-optimize (optional for hyperparameter optimization)

You can use pip to install dependencies:

    pip install -r requirements.txt

Note that this will install tensorflow without GPU support. Checkout the [tensorflow documentation](https://www.tensorflow.org/install/pip) to install with GPU support

Invoke
------

To start training:

    pinky --config <config_filename> --train

You can dump your examples to TFRecordDatasets to accelerate io operations:

    pinky --config <config_filename> --write <new_config_filename>

and use the newly created config file to run `--train`

Invoke pinky with `--debug` to enable keep track of weight matrices in
[tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
