
```



import tensorflow as tf
import time

global vocab
vocab = [ "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]

class CycleOne(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(CycleOne, self).__init__()
		self.num_outputs = num_outputs
		
	def build(self, input_shape):
		min_size_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1, seed=None)
		self.kernel = self.add_weight(shape=[int(input_shape[-1]),
						self.num_outputs],
                        initializer = min_size_init,
                        trainable=True)
		
		array = tf.ones([ self.num_outputs, 1 ], dtype=tf.int64)
		
		self.kernel = array
		self.kernel = tf.cast( self.kernel, dtype=tf.int64 )
		
	def call(self, inputs):
		array = tf.ones([self.num_outputs, 1], dtype=tf.int64) * 6
		array = tf.matmul(self.kernel, inputs) + array
		array = array % 26
		
		return array
		
class CycleTwo(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(CycleTwo, self).__init__()
		self.num_outputs = num_outputs
		
	def build(self, input_shape):
		min_size_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1, seed=None)
		self.kernel = self.add_weight(shape=[int(input_shape[-1]),
						self.num_outputs],
                        initializer = min_size_init,
                        trainable=True)
		
		array = tf.ones([ self.num_outputs, 1 ], dtype=tf.int64)
		
		self.kernel = array
		self.kernel = tf.cast( self.kernel, dtype=tf.int64 )
		
	def call(self, inputs):
		array = tf.constant([7, 7, 11, 22, 31], dtype=tf.int64)
		paddedsize=tf.constant([[0, self.num_outputs - array.shape[0]]])
		paddings = paddedsize
		array = tf.pad(array, paddings, mode='CONSTANT', constant_values=0)
		
		array = tf.matmul(self.kernel, inputs) + array
		array = array % 26
		array = tf.cast( array, dtype=tf.float32 )
		array = tf.where( tf.math.equal( tf.zeros([ array.shape[0], array.shape[1] ], dtype=tf.float32), array ), 26, array ).numpy()[0]
		array = tf.cast( array, dtype=tf.int64 )
		
		return array


def convert_constant_text_to_number_array( constant_text ):
	global vocab
	
	layer = tf.keras.layers.StringLookup(vocabulary=vocab)
	
	sequences_mapping_string = layer(constant_text)
	sequences_mapping_string = tf.constant( sequences_mapping_string, shape=(1, constant_text.shape[0]) )
	
	return sequences_mapping_string
	
def convert_text_to_constant_text( text, paddedsize=12 ):
	input_text = list(text)
	input_text = tf.constant( input_text )

	paddedsize=tf.constant([[0, paddedsize - len(text)]])
	paddings = paddedsize
	ans = tf.pad(input_text, paddings, mode='CONSTANT', constant_values="!")

	return ans

def decode_from_vocab( text_sequence ):
	global vocab
	
	decoder = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode="int", invert=True)
	ans = decoder(text_sequence)

	return ans
	
def feedforward( input ):
	array_text = convert_text_to_constant_text( input,  len(input) )
	array_text_as_sequence = convert_constant_text_to_number_array( array_text )
	
	layer = CycleOne( len(input) )
	data = layer(array_text_as_sequence)

	text = decode_from_vocab( data )

	ans = ""
	for item in text[0] :
		ans = ans + item
	
	return str(ans.numpy())[2:-1]
	
def feedforward_two( input ):
	array_text = convert_text_to_constant_text( input,  len(input) )
	array_text_as_sequence = convert_constant_text_to_number_array( array_text )
	
	layer = CycleTwo( len(input) )
	data = layer(array_text_as_sequence)

	text = decode_from_vocab( data )

	ans = ""
	for item in text :
		ans = ans + item
	
	return str(ans.numpy())[2:-1]


print("")
input = "hello"
output = feedforward( input )
print( "input: " + input )
print( "output: " + output )

input = output
output = feedforward_two( input )
print( "input: " + input )
print( "output: " + output )

print( convert_constant_text_to_number_array( convert_text_to_constant_text( "hello" ) ) )
# tf.Tensor([[ 8  5 12 12 15  0  0  0  0  0  0  0]], shape=(1, 12), dtype=int64)
print( convert_constant_text_to_number_array( convert_text_to_constant_text( "nkruu" ) ) )
# tf.Tensor([[14 11 18 21 21  0  0  0  0  0  0  0]], shape=(1, 12), dtype=int64)
print( convert_constant_text_to_number_array( convert_text_to_constant_text( "urcnz" ) ) )
# tf.Tensor([[21 18  3 17 26  0  0  0  0  0  0  0]], shape=(1, 12), dtype=int64)


```
