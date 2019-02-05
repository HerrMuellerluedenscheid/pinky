Example
=======


Generate test data
------------------


    cd data
    colosseo fill
    cd ..

This will download a greens functions database of about 1.5 GB. After downloading
a dataset will be generated. This should take few minutes.


Train Pinky
-----------


The input file `pinky.config` is ready to be trained:

    pinky --config pinky.config --train

Training this way will be both, memory inefficient as well as slow. Pinky
provides a shortcut to dump all examples into a tensorflow compatible tfrecord
file:

    pinky --config pinky.config --write-tfrecord pinky

Two files with file ending `tfrecord` will be generated conaining features and
labels of the training and the evaluation data generator defined in your
`pinky.config` file. Furthermore, another config file if generated,
`pinky.tf.config` which you can now use to tune the model.

Note that variables with a prepending underscore (`_`) are generated
programmatically and are not meant to be modified.
