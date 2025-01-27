# code follows tutorial from https://www.tensorflow.org/tutorials/generative/style_transfer
import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import numpy as np

from utils import load_img, tensor_to_image, vgg_layers, clip_0_1, style_content_loss
from style_content_model import StyleContentModel


content_path = "img/in1.png"
style_path = "img/gogh.png"

content_image = load_img(content_path)
style_image = load_img(style_path)


# vgg without fc and softmax layers
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')


content_layers = ['block5_conv2'] 

"""1
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
"""
"""2
style_layers = ['block1_conv1',
                'block1_conv2',
                'block2_conv1', 
                'block2_conv2', 
                'block3_conv1']
"""
#"""3
style_layers = ['block4_conv4',
                'block5_conv1',
                'block5_conv2', 
                'block5_conv3', 
                'block5_conv4']
#"""
"""4
style_layers = ['block1_conv1',
                'block1_conv2',
                'block2_conv1', 
                'block5_conv3', 
                'block5_conv4']
"""

"""5
style_layers = ['block1_conv1',
                'block1_conv2']
"""

"""6
style_layers = ['block1_conv1',
                'block2_conv1']
"""

"""7
style_layers = ['block5_conv3',
                'block5_conv4']
"""

"""8
style_layers = ['block4_conv4',
                'block5_conv4']
"""

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
# influence of style and content
weights = [1e-1, 1e5] #1e-2 #1e4
total_variation_weight=30

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, style_targets, weights[0], num_style_layers, content_targets, weights[0], num_content_layers)
    loss += total_variation_weight*tf.image.total_variation(image)

  # create new image
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

epochs = 10
steps_per_epoch = 100

for epoch in range(epochs):
  for step in range(steps_per_epoch):
    train_step(image)
image1 = tensor_to_image(image)
image1.save(f"imgs/output_gogh_-15_3.png")
