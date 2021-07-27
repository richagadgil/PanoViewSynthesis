import tensorflow as tf
import sys
sys.path.append('/Users/johnwaidhofer/Desktop/Code/SummerResearch/PanoViewSynthesis/single_view_mpi')
from single_view_mpi import nets
from single_view_mpi.libs import mpi
import matplotlib.pyplot as plt


input = tf.keras.Input(shape=(None, None, 3))
output = nets.mpi_from_image(input)

model = tf.keras.Model(inputs=input, outputs=output)
print('Model created.')
# Our full model, trained on RealEstate10K.
model.load_weights('single_view_mpi_full_keras/single_view_mpi_keras_weights')
print('Weights loaded.')


plt.rcParams["figure.figsize"] = (20, 10)

# Input image
inputfile = 'cube_dir/apartment_1/00/spherical_panorama/sphere_00_frame.png'
layer_output = 'cube_dir/apartment_1/spherical_layer'
predicted_output = 'cube_dir/apartment_1/predicted_spherical_disparity.png'
input_rgb = tf.image.decode_image(tf.io.read_file(inputfile), dtype=tf.float32)

# Generate MPI
layers = model(input_rgb[tf.newaxis])[0]
depths = mpi.make_depths(1.0, 100.0, 32).numpy()

# Layers is now a tensor of shape [L, H, W, 4].
# This represents an MPI with L layers, each of height H and width W, and
# each with an RGB+Alpha 4-channel image.

# Depths is a tensor of shape [L] which gives the depths of the L layers.

# Display layer images
for i in range(32):
    plt.imsave(f"{layer_output}/layer_{i}.png", layers[i], cmap="gray")
#   plt.subplot(4, 8, i+1)
#   plt.imshow(layers[i])
#   plt.axis('off')
#   plt.title('Layer %d' % i, loc='left')
# plt.show()

# Display computed disparity
disparity = mpi.disparity_from_layers(layers, depths)
plt.imsave(predicted_output, disparity[..., 0], cmap="gray")


