import torch
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"model saved to: '{path}'")

def load_model(model, path, strict=True):
    new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(new_state_dict, strict=strict)
    print(f"Model loaded from: '{path}'")

def file2model(path, mod=False, auto=True):
    new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    hidden_size = new_state_dict['lstm.weight_hh_l0'].size(1)
    vocab_size = new_state_dict['embedding_matrix.weight'].size(0)
    embedding_size = new_state_dict['embedding_matrix.weight'].size(1)
    
    train_states   = ("h0" in new_state_dict and "c0" in new_state_dict)
    tied_embedding = ("out_b" in new_state_dict)
    if mod:
        model_class = LMEntity_mod
        print('Using class: LMEntity_mod')
    else:
        model_class = LMEntity
        print('Using class: LMEntity')
        
    if auto:
        model_class = eval(path.split('/')[-1].split('_')[0])
        
    model = model_class(hidden_size=hidden_size, 
                     vocab_size=vocab_size, 
                     embedding_size=embedding_size,
                     train_states=train_states,
                     tied_embedding=tied_embedding).to(device)
    model.load_state_dict(new_state_dict)
    print(f"Model loaded from: '{path}'")
    print(f'hidden_size: {hidden_size}\nvocab_size: {vocab_size}\nembedding_size: {embedding_size}\ntrained_states: {train_states}\ntied_embedding: {tied_embedding}')
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

