
from math import sqrt
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.layers import Layer
from keras.layers import Add
from keras import backend
from keras.models import load_model
from matplotlib import pyplot

class PixelNormalization(Layer):
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)

def call(self, inputs):
	values = inputs**2.0
	mean_values = backend.mean(values, axis=-1, keepdims=True)
	mean_values += 1.0e-8
	l2 = backend.sqrt(mean_values)
	normalized = inputs / l2
	return normalized

def compute_output_shape(self, input_shape):
	return input_shape

class MinibatchStdev(Layer):
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)

def call(self, inputs):
	mean = backend.mean(inputs, axis=0, keepdims=True)
	squ_diffs = backend.square(inputs - mean)
	mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
	mean_sq_diff += 1e-8
	stdev = backend.sqrt(mean_sq_diff)
	mean_pix = backend.mean(stdev, keepdims=True)
	shape = backend.shape(inputs)
	output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
	combined = backend.concatenate([inputs, output], axis=-1)
	return combined

def compute_output_shape(self, input_shape):
	input_shape = list(input_shape)
	input_shape[-1] += 1
	return tuple(input_shape)

class WeightedSum(Add):
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = backend.variable(alpha, name='ws_alpha')

def _merge_function(self, inputs):
	assert (len(inputs) == 2)
	output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
	return output

def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

def plot_generated(images, n_images, tempcreate):
	square = int(sqrt(n_images))+1
	images = (images - images.min()) / (images.max() - images.min())
	for i in range(n_images):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(images[i])
	pyplot.savefig('static/ESRGAN/LR/PicGenerated{}.png'.format(tempcreate+1), transparent=True, bbox_inches = 'tight')

def castpic(modelname):
	for tempcreate in range(4):
		cust = {'PixelNormalization': PixelNormalization, 'MinibatchStdev': MinibatchStdev, 'WeightedSum': WeightedSum}
		model = load_model(modelname, cust)
		latent_dim = 100
		n_images = 1   #  number of images to generate
		latent_points = generate_latent_points(latent_dim, n_images)
		X  = model.predict(latent_points)
		plot_generated(X, n_images, tempcreate)

""" 
modelname = 'model/forestAice_generator648.h5'
cust = {'PixelNormalization': PixelNormalization, 'MinibatchStdev': MinibatchStdev, 'WeightedSum': WeightedSum}
model = load_model(modelname, cust)
latent_dim = 100
n_images = 1   #  number of images to generate
latent_points = generate_latent_points(latent_dim, n_images)
X  = model.predict(latent_points)
plot_generated(X, n_images)
"""
