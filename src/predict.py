import tensorflow as tf


import tensorflow as tf
from tensorflow.compat.v1.lookup import StaticHashTable, KeyValueTensorInitializer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import argparse, joblib

import pandas as pd


# set up argument parsing (make sure these match those in config.yml)
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True)
args = parser.parse_args()

# READ DATA
data = pd.read_csv(args.infile)

def seq_to_number(seq_, max_length):
    seq_ = pd.DataFrame(np.array([seq_]).T)
    seq_ = [list(i[0]) for i in seq_.values]
    seq_ = pad_sequences(seq_, dtype='str', maxlen=max_length, padding='post', truncating='post')
    seq_ = table.lookup(tf.constant(seq_, dtype = tf.string))
    return seq_




table = StaticHashTable(
            initializer=KeyValueTensorInitializer(
                keys=tf.constant(['0', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'U', 'O', '\n'], dtype = tf.string),
                values=tf.constant([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24], dtype = tf.int32),
                key_dtype= tf.string,
                value_dtype=tf.int32,
            ),
            default_value=tf.constant(1, dtype = tf.int32),
            name="class"
        )

#Parametars 
MAX_LENGTH = 150


NUM_LAYERS= 1
NUM_HEADS= 2 
D_MODEL = 32 * NUM_HEADS 
DFF = 2 * 32 * NUM_HEADS

INPUT_VOCAB_SIZE = 25

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

  


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)


    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)



    
def create_model():
    inputs_1 = layers.Input((MAX_LENGTH,), dtype=tf.int64)
    inputs_2 = layers.Input((MAX_LENGTH,), dtype=tf.int64)

    Bert_1 = Encoder(num_layers = NUM_LAYERS, d_model = D_MODEL, num_heads = NUM_HEADS,
                         dff = DFF, input_vocab_size = INPUT_VOCAB_SIZE,
                         maximum_position_encoding = MAX_LENGTH)(inputs_1, training=True, mask=None)
    
    Bert_2 = Encoder(num_layers = NUM_LAYERS, d_model = D_MODEL, num_heads = NUM_HEADS,
                         dff = DFF, input_vocab_size = INPUT_VOCAB_SIZE,
                         maximum_position_encoding = MAX_LENGTH)(inputs_2, training=True, mask=None)

    concatenated_ = tf.keras.layers.Concatenate(axis = -2)([Bert_1, Bert_2])
    dense_ = tf.keras.layers.Conv1D(filters = 4, kernel_size = 1)(concatenated_)  
    dense_ = tf.keras.layers.Conv1D(filters = 2, kernel_size = 1)(dense_)
    dense_ = tf.keras.layers.Conv1D(filters = 1, kernel_size = 1)(dense_)

    fllatened_ = tf.keras.layers.Flatten()(dense_)
    prediction_ = tf.keras.layers.Dense(units = 1, activation="sigmoid")(fllatened_)

    clasiffier = tf.keras.models.Model([inputs_1, inputs_2],prediction_, name="classifier")

    return clasiffier

Clasiffier = create_model()


optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005)
Clasiffier.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", "AUC"])

x_1 = seq_to_number(list(data.Hchain), MAX_LENGTH) #ovo triba prominit ui data.Hchain, data.Lchain
x_2 = seq_to_number(list(data.Lchain), MAX_LENGTH)

Clasiffier.load_weights('src/Model_w.h5')

y_pred = Clasiffier.predict([x_1, x_2])

# SAVE PREDICTIONS WITH THE COLUMN NAME prediction IN THE FILE predictions.csv
pd.DataFrame(y_pred[:,0], columns=['prediction']).to_csv("predictions.csv", index=False)
print(pd.DataFrame(y_pred[:,0], columns=['prediction']))
