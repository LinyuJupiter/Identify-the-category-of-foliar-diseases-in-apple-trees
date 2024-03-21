import tensorflow as tf


class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, scale=30.0, margin=0.5, reduction=tf.keras.losses.Reduction.AUTO, name='arc_loss'):
        super(ArcLoss, self).__init__(reduction=reduction, name=name)
        self.scale = scale
        self.margin = margin

    def call(self, y_true, y_pred):
        # Cast y_true to float32
        y_true = tf.cast(y_true, tf.float32)

        # Normalize vectors
        y_true = tf.math.l2_normalize(y_true, axis=1)
        y_pred = tf.math.l2_normalize(y_pred, axis=1)

        # Compute cosine similarity
        cos_sim = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)

        # Compute the arc margin
        arc_margin = tf.math.acos(cos_sim - self.margin)

        # Compute the final loss
        loss = tf.reduce_mean(tf.square(tf.maximum(0.0, arc_margin)))

        return self.scale * loss
