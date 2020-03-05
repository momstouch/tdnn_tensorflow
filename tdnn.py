import tensorflow as tf
import numpy as np


def renorm_layer(
        inputs,
        input_dim
        ):
    #
    # original code from Kaldi's NormalizeComponent
    # y = x * (sqrt(dim(x)) * target-rms) / |x|
    # for details, http://kaldi-asr.org/doc/nnet-normalize-component_8h_source.html
    #

    sqrt_output_dim = tf.sqrt(x = float(input_dim))

    norm = tf.cast(
            x = tf.reciprocal(
                x = tf.norm(
                    tensor = inputs,
                    ord = 2,
                    axis = 2
                    )
                ),
            dtype = tf.float32
            )
    norm = tf.expand_dims(
            input = norm,
            axis = 2
            )
    # norm.shape: [batch_size, sequence_length, input_dim]

    sqrt_output_dims_mat = tf.reshape(
            tensor = sqrt_output_dim,
            shape = [1, 1]
            )
    sqrt_output_dims_mat = tf.multiply(
            x = norm,
            y = sqrt_output_dims_mat
            )

    scale_value = tf.multiply(
            x = inputs,
            y = sqrt_output_dims_mat
            )

    return scale_value


def tdnn(
        inputs, # [batch_size, sequence_length, feature_dims]
        context : list,
        output_dim : int,
        layer_name : str,
        input_dim : int = None,
        full_context = False,
        pnorm_name : str = None,
        renorm_name : str = None,
        weights = None,
        use_bias : bool = True,
        bias = None,
        trainable : bool = True
        ):
    #
    # the original code from https://github.com/SiddGururani/Pytorch-TDNN
    #

    def check_valid_context(context : list):
        assert context[0] <= context[-1], \
                "Input tensor dimensionality is incorrect. Should be a 3D tensor"


    def get_kernel_width(context : list, full_context : bool):
        if full_context:
            context = list(range(context[0], context[-1] + 1))
        return len(context), context


    def get_valid_steps(context : list, input_sequence_length : int):
        start = 0 if context[0] >= 0 else  -1 * context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return tf.range(start=start, limit=end)


    def is_full_context(context : list):
        for idx, c in enumerate(context):
            try:
                if c + 1 != context[idx + 1]:
                    return False
            except IndexError:
                return True


    input_shape = tf.shape(inputs)
    sequence_length = input_shape[1]
    check_valid_context(context)
    kernel_width, context = get_kernel_width(context, full_context)
    valid_steps = get_valid_steps(context, sequence_length)

    context = [context] * 1
    context = tf.tile(input = context, multiples = [tf.size(valid_steps), 1])
    valid_steps = tf.expand_dims(valid_steps, 1)

    selected_indices = context + valid_steps

    # features.shape: [batch, sequence_length_of_output, kernel, feats_dims]
    # reshape -> [batch, sequence_length_of_output, kernel * feats_dims]
    # e.g.,   -> [batch, length, channels]
    features = tf.gather(
            params = inputs,
            indices = selected_indices,
            axis = 1
            )
    f_shape = tf.shape(features)
    features = tf.reshape(
            tensor = features,
            shape = [f_shape[0], f_shape[1], -1]
            )

    if not weights:
        weights = tf.contrib.layers.xavier_initializer()

    #regularizer = None if not trainable else \
    #        tf.contrib.layers.l2_regularizer(scale = 0.01)

    out = tf.layers.conv1d(
            inputs = features,
            filters = output_dim,
            kernel_size = 1,
            strides = 1,
            data_format = "channels_last",
            use_bias = use_bias,
            name = layer_name,
            kernel_initializer = weights,
            kernel_regularizer = None,
            bias_initializer = bias,
            trainable = trainable
            )

    if pnorm_name:
        out_shape = tf.shape(out)
        out = tf.reshape(
                tensor = out,
                shape = [out_shape[0], out_shape[1], input_dim, -1]
                )
        out = tf.norm(
                tensor = out,
                ord = 2,
                axis = 3,
                name = pnorm_name
                )

    if renorm_name:
        out = renorm_layer(
                inputs = out,
                input_dim = input_dim
                )

    return out
