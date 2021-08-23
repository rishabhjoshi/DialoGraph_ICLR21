
# -*- coding: utf-8 -*-
'''
Created on : Friday 19 Jun, 2020 : 18:46:50
Last Modified : Friday 19 Jun, 2020 : 18:52:46

@author       : Rishabh Joshi
Institute     : Carnegie Mellon University
'''
import sys
sys.path.append('../')
from helper import *
### Transformer ###
class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	#def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
	def __init__(self, decoder, tgt_embed, d_model, vocab, config, feature_weight):
            super(EncoderDecoder, self).__init__()
            #self.encoder = encoder
            self.decoder = decoder
            #self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            #self.generator = generator
            self.proj = nn.Linear(d_model, vocab)
            try:
                if 'only_hidden' in config:
                    self.only_hidden = config.only_hidden
                else:
                    self.only_hidden = False
            except:
                self.only_hidden = False
		
	#def forward(self, src, tgt, src_mask, tgt_mask):
	def forward(self, feats, utt_mask, ratios=None, return_attn_weights=False):
		'''
		feats : num_conv x num_utt x num_feats		# num_feats is 1 for clustered data
		utt_mask : num_conv x num_utt
		ratios: num_conv x 1
		return_attn_weights: boolean
		'''
		"Take in and process masked src and target sequences."
		to_augment = torch.zeros((feats.shape[0], feats.shape[1], 1)).to(feats.device) # this is agumented to make num of heads div
		feats = torch.cat((feats, to_augment), dim = 2)
		assert feats.shape[2] % 2 == 0
		num_conv = feats.shape[0]
		num_utt  = feats.shape[1]
		num_feats = feats.shape[2]
		# return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
		memory = None
		src_mask = None
		# Have to create tgt and tgt_mask
		tgt = feats
		#tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = (utt_mask == 1).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(tgt_mask.size(-1)).type_as(tgt_mask.data))

		outputs, layer_attn_weights = self.decode(memory, src_mask, tgt, tgt_mask, return_attn_weights)
		if self.only_hidden:
			return outputs
		all_logits  = self.proj(outputs)

		# is_valid = (utt_mask[:, 1:] == 1).unsqueeze(2)			# num_conv x num_utt-1 x 1
		# logitloss = self.logitcriterion(all_logits[:, :-1, :], feats[:, 1:, :])
		# den = torch.sum(is_valid.view(-1))				# for valid conv
		# logitloss = torch.sum(logitloss * is_valid.float()) / (den * num_feats)
		# return logitloss, 0.0, all_logits, 0.0, layer_attn_weights
		return all_logits, layer_attn_weights
		 #return self.decode(None, src_mask, tgt, tgt_mask)
	
	# def encode(self, src, src_mask):
	# 	return self.encoder(self.src_embed(src), src_mask)
	
	def decode(self, memory, src_mask, tgt, tgt_mask, return_attn_weights = False):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, return_attn_weights)

# Encoder Decoder Stacks

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

# Decoder
class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask, return_attn_weights = False):
		layer_attn_weights = torch.zeros((len(self.layers), x.shape[0], 2, x.shape[1], x.shape[1])).to(x.device)
		i = 0
		for layer in self.layers:
			if return_attn_weights:
				x, attn_weights = layer(x, memory, src_mask, tgt_mask, return_attn_weights)
				layer_attn_weights[i] = attn_weights
			else:
				x = layer(x, memory, src_mask, tgt_mask)
			i += 1
				#layer_attn_weights.append(attn_weights)
		return self.norm(x), layer_attn_weights

class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn						# MultiheadedAttention (h = 2, d_model = 22)
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask, return_attn_weights = False):
		"Follow Figure 1 (right) for connections."
		m = memory
		if return_attn_weights:
			attention_weights = self.self_attn(x, x, x, tgt_mask, return_attn_weights)					# 6 x 16 x 22
		# import pdb; pdb.set_trace()
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		#x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		if return_attn_weights:
			return self.sublayer[2](x, self.feed_forward), attention_weights
		return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

# Attention
def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None, return_attn_weights = False):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		#pdb.set_trace()								# earlier shape : 6 x 16 x 22
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear.	old x: 6 x 2 x 16 x 11 -> 6 x 16 x 2 x 11 
		x = x.transpose(1, 2).contiguous() \
				.view(nbatches, -1, self.h * self.d_k)		# newx : 6 x 16 x 22
				
		# Attention has shape					num_conv x 2 x num_utt x num_utt
		if return_attn_weights:
			return self.attn

		return self.linears[-1](x)

# Position wise feed forward
class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

# Embeddings
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return x
		#return self.lut(x) * math.sqrt(self.d_model)

# Positional encoding
class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0., max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., d_model, 2) *
							 -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], 
						 requires_grad=False)
		return self.dropout(x)

# Make Model
#def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
def make_model(config, tgt_vocab=22, N=6, d_model=22, d_ff=2048, h=2, dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	d_model += 1 # will add 1 col of zeros to data
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	feature_weights = None
	model = EncoderDecoder(
		#Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		#nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		#Generator(d_model, tgt_vocab), 
		d_model, 
		tgt_vocab,
		config,
		feature_weights)
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model
