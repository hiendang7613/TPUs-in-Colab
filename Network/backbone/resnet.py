import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv3(inputs)
            res = self.bn3(res, training=training)
        x += res
        x = self.relu(x)
        return x


class Bottleneck(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same', strides=strides)
        self.bn4 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv4(inputs)
            res = self.bn4(res, training=training)
        x += res
        x = self.relu(x)
        return x


class ResNet_v1(tf.keras.Model):
    def __init__(self, Block=Bottleneck, layers=(3, 4, 6, 3), include_top=True, embedding_size=512):
        super(ResNet_v1, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = tf.keras.Sequential([Block(filters=64, strides=(1, 1)) for _ in range(layers[0])])
        self.blocks2 = tf.keras.Sequential(
            [Block(filters=128, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[1])])
        self.blocks3 = tf.keras.Sequential(
            [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])])
        self.blocks4 = tf.keras.Sequential(
            [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])])
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = None
        if include_top:
            self.dense = tf.keras.layers.Dense(embedding_size)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.blocks1(x, training=training)
        x = self.blocks2(x, training=training)
        x = self.blocks3(x, training=training)
        x = self.blocks4(x, training=training)
        x = self.globalpool(x)
        if self.dense is not None:
            x = self.dense(x)

        return x


class ResNet_v1_18(ResNet_v1):
    def __init__(self, include_top=True, embedding_size=512):
        super(ResNet_v1_18, self).__init__(Block=BasicBlock, layers=(2, 2, 2, 2), include_top=include_top,
                                           embedding_size=embedding_size)


class ResNet_v1_34(ResNet_v1):
    def __init__(self, include_top=True, embedding_size=512):
        super(ResNet_v1_34, self).__init__(Block=BasicBlock, layers=(3, 4, 6, 3), include_top=include_top,
                                           embedding_size=embedding_size)


class ResNet_v1_50(ResNet_v1):
    def __init__(self, include_top=True, embedding_size=512):
        super(ResNet_v1_50, self).__init__(Block=Bottleneck, layers=(3, 4, 6, 3), include_top=include_top,
                                           embedding_size=embedding_size)


class ResNet_v1_101(ResNet_v1):
    def __init__(self, include_top=True, embedding_size=512):
        super(ResNet_v1_101, self).__init__(Block=Bottleneck, layers=(3, 4, 23, 3), include_top=include_top,
                                            embedding_size=embedding_size)


class ResNet_v1_152(ResNet_v1):
    def __init__(self, include_top=True, embedding_size=512):
        super(ResNet_v1_152, self).__init__(Block=Bottleneck, layers=(3, 8, 36, 3), include_top=include_top,
                                            embedding_size=embedding_size)
