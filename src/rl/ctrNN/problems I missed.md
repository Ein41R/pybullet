### Things Ive learned:

Calling .detatch() when we backpropagate, while having parallel paths in the graph. Having two backward passes result into a error. e.g. double dependency of
'''
loss = loss_fn(pred, target)
pred_dec_loss = decoder(pred, y)
'''

Using fixed size positional embedding throughout a sequence imposing a hard limit for context length.
''''''


#TODO:
- there is some kind of model collapse going on. I get a string of \x00 as result.
The target encoder is a EMA copy of the 

The focused check confirms gradients are connected, but it also exposes a real collapse mechanism: the predictor’s decoder loss backpropagates into the decoder, and then the target-decoder loss is accumulated on top of those gradients before the decoder step. The probe can therefore learn to decode whatever shortcut the predictor emits, instead of remaining a fixed test of whether the prediction is meaningful. I’ll fix that wiring and add explicit target-offset embeddings, which the current predictor lacks.