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