#   The number of squares occupying:
#       1. one rank and file (15)
#       2. both "complete" diagonals (14)
#       3. the full "reach" of a knight (8)
FILTER_LEN = 37

class ChessConv(keras.layers.Layer):
    def __init__(self, units):
        super(ChessConv, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(FILTER_LEN, self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        verts = tf.multiply(self.w[:8, :], inputs
