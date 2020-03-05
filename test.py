import tensorflow as tf
from tensorflow.python.framework import ops

import tdnn as TDNN

minibatch_size = 1
pnorm_input_dim = 128
pnorm_output_dim = 64

input_sequence_length = 13 + 1 + 9 # prev + center + post frames
input_data_dims = 120

tdnn_names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
tdnn_context = [[-2, -1, 0, 1, 2], [-1, 2], [-3, 3], [-7, 2], [0]]

layer_dict = {}

end_layer = layer_dict["input_layer"] = tf.Variable(
        tf.random_uniform([minibatch_size, input_sequence_length, input_data_dims],
            -1.0,
            1.0
            )
        )
print(end_layer)

for idx, (layer_name, context) in enumerate(zip(tdnn_names, tdnn_context)):
    with ops.name_scope(name = layer_name):
        pnorm_name = "pnorm" + str(idx + 1)
        renorm_name = "renorm" + str(idx + 1)

        end_layer = layer_dict[layer_name] = TDNN.tdnn(
                inputs = end_layer,
                context = context,
                input_dim = pnorm_output_dim,
                output_dim = pnorm_input_dim,
                layer_name = layer_name,
                pnorm_name = pnorm_name,
                renorm_name = renorm_name
                )

for key, val in layer_dict.items():
    print(key, val)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    logits = sess.run(end_layer)
    print(logits)
