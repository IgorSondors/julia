import click
import tensorflow as tf



class Julia(tf.keras.Model):
    def __init__(self, model_path):
        super().__init__()

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def preprocess(self, inputs):
        return inputs

    def postprocess(self, outputs):
        return tf.math.reduce_mean(outputs), outputs

    def get_9_superblocks(self, uimage, blocksize):
        h = tf.shape(uimage)[1]
        w = tf.shape(uimage)[2]
        hb = h - blocksize
        wb = w - blocksize

        blocks_0 = tf.expand_dims(uimage[:, 0:blocksize, 0:blocksize, :], axis=1)
        blocks_1 = tf.expand_dims(uimage[:, hb:h, 0:blocksize, :], axis=1)
        blocks_2 = tf.expand_dims(uimage[:, 0:blocksize, wb:w, :], axis=1)
        blocks_3 = tf.expand_dims(uimage[:, hb:h, wb:w, :], axis=1)
        blocks_4 = tf.expand_dims(uimage[:, (hb) // 2:(h + blocksize) // 2, 0:blocksize, :], axis=1)
        blocks_5 = tf.expand_dims(uimage[:, 0:blocksize, (wb) // 2:(w + blocksize) // 2, :], axis=1)
        blocks_6 = tf.expand_dims(uimage[:, (hb) // 2:(h + blocksize) // 2, (wb) // 2:(w + blocksize) // 2, :], axis=1)
        blocks_7 = tf.expand_dims(uimage[:, (hb) // 2:(h + blocksize) // 2, wb:w, :], axis=1)
        blocks_8 = tf.expand_dims(uimage[:, hb:h, (wb) // 2:(w + blocksize) // 2, :], axis=1)

        blocks = tf.concat([blocks_0,
                            blocks_1,
                            blocks_2,
                            blocks_3,
                            blocks_4,
                            blocks_5,
                            blocks_6,
                            blocks_7,
                            blocks_8,
                            ], axis=1)
        return blocks

    def decode_and_extract_patches(
            self,
            inputs
    ):  

        bitstring = inputs[0]
        image = tf.io.decode_image(bitstring, channels=3)
        image = tf.expand_dims(image, axis=0)
        patches = self.get_9_superblocks(image, 128)
        patches = tf.reshape(patches, [9, 128, 128, 3])
        return patches

    def __call__(self, inputs, training):
        if inputs.dtype == tf.string:
            inputs = self.decode_and_extract_patches(inputs)

        if inputs.dtype == tf.uint8:
            inputs = tf.cast(inputs, tf.float32)

        inputs = self.preprocess(inputs)

        outputs = self.model(inputs)

        confidence, all_confidences = self.postprocess(outputs)

        return {"confidence": confidence, "all_confidences": all_confidences}



@click.command()
@click.option('--model_path', type=str, default='/home/sondors/Documents/export_spoof/c3ae-128-dct_ugreen-005-0.1076.h5',help='папка с исходными файлами')
@click.option('--export_dir', type=str, default='/home/sondors/Documents/export_spoof/julia', help='целевая папка')
def fmain(model_path, export_dir):


    julia = Julia(model_path=model_path)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string, name="images_bytes")])
    def serving_default(images_bytes):
        return julia(images_bytes, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.uint8, name="images_pixels")])
    def predicting_default(images_pixels):
        return julia(images_pixels, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32, name="preprocessed_images_pixels")])
    def constructing_default(preprocessed_images_pixels):
        return julia(preprocessed_images_pixels, training=False)

    signatures = {}
    signatures["serving_default"] = serving_default.get_concrete_function()
    signatures["predicting_default"] = predicting_default.get_concrete_function()
    signatures["constructing_default"] = constructing_default.get_concrete_function()

    tf.saved_model.save(julia, export_dir=export_dir, signatures=signatures)

if __name__ == "__main__":
    fmain()