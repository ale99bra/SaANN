from . import backend as BE
from .losses import cross_entropy_logits, cross_entropy_logits_der

def train_transformer_step(model, optimizer, tokens, targets):

    #forward pass
    logits = model.forward(tokens)

    #loss
    loss = cross_entropy_logits(logits=logits, target_ids=targets)

    #backward pass
    d_logits = cross_entropy_logits_der(logits=logits, target_ids=targets)
    model.backward(d_logits)   

    #gradient clipping
    for name, p in optimizer.params.items():
        if name.startswith("d"):
            BE.xp.clip(p, -1.0, 1.0, out=p)

    #optimizer
    optimizer.step()

    #reset
    model.zero_grad()

    return loss

def train_transformer(model, optimizer, tokens, targers, steps = 1000):
    for step in range(1, steps+1):
        loss = train_transformer_step(model=model, optimizer=optimizer, tokens=tokens, targets=targers)

    perc = 0
    if step == 1:
        print(f"Loss at first step: {loss:.4f}")
    elif step % steps/10 == 0:
        perc += 1
        print(f"At {perc*10}% - Loss: {loss:.4f}")