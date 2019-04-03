Example
=======

Following this example you will:

- Download of a GF database 
- Compute synthetic waveform data and P phase onset times for a given scenario (synth. event catalog)
- Train a convolutional neural network to locate the events

Generate test data
------------------


    cd data
    colosseo fill
    cd ..

This will download a greens functions database of about 1.5 GB. After downloading
a dataset will be generated. This should take few minutes.

To adjust settings for the scenario you can modify data/scenario.yml. For better training, the number of events (nevents) can be increased.


Train Pinky
-----------


The input file `pinky.config` is ready to be trained:

    pinky --config pinky.config --train

In the current settings, the network has 2 convolutional layers of width/ height/ number of filters 10/1/32 and 3/3/64, followed by a dense layer with 64 neurons and an output layer with 3 neurons (x,y,z coordinates).

Data is filtered between 1 and 20 Hz and a batch size of 20 examples is used.

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


Tensorboard
-----------



	tensorboard --logdir summary --port 8080




