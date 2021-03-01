import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *

from collections import defaultdict 
import time, sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}\n')

def run_nlm(model, corpus, optimizer=None, epochs=1, eval_corpus=None, status_interval=25, str_pattern='{}_{}_epoch_{}.pkl', rz_amplifier=10):
    if optimizer:
        print(f'Start training {model.__class__.__name__} with {model.lstm.hidden_size} units for {epochs} epoch(s).')
    entity_offset = 1

    for epoch in range(1, epochs+1):
        X_epoch_loss, E_epoch_loss, R_epoch_loss, L_epoch_loss = 0, 0, 0, 0
        epoch_tokens, epoch_r_div, epoch_l_div, epoch_e_div = 0, 0, 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_R = 0
        r_true_positive  = 0
        r_false_positive = 0
        for i_doc, doc in enumerate(corpus.gen()):
            model.clear_entities()
            current_entity = model.get_new_entity()
            assert model.entities.size(0) == 1 # Only 1 empty entity
            h_t, states = model.forward_rnn(doc.X[0])
            h_t = h_t.squeeze(0)
            X_loss = torch.tensor(0, dtype=torch.float, device=device)
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            R_loss = torch.tensor(0, dtype=torch.float, device=device)
            L_loss = torch.tensor(0, dtype=torch.float, device=device)
            
            r_div = 0
            l_div = 0
            e_div = 0


            
            for t in range(doc.X.size(0)-1):                
                next_X = doc.X[t+1]
                next_E = doc.E[t+1] - entity_offset
                next_R = doc.R[t+1]
                next_L = doc.L[t+1]
                
                # Start Paper
                current_L = doc.L[t]
                if current_L == 1:
                    # 1. Not continue EM
                    # last L equals 1, not continuing entity mention
                    # predict R
                    
                    pred_R = model.predict_type(h_t)
                    r_current_loss = torch.nn.functional.cross_entropy(pred_R, next_R.view(-1))

                    
                    r_div  += 1

                    
                    if next_R == 1:
                        count_R += 1
                        if pred_R.argmax():
                            # both True - correct pred
                            r_true_positive  += 1
                            R_loss += r_current_loss
                        else:
                            # wrong False pred
                            # extra loss
                            R_loss += r_current_loss * rz_amplifier
                        
                        # If there is no embedding - create one
                        # Select the entity
                        pred_E = model.predict_entity(h_t, t)
                        count_E += 1
                        count_R += 1
                        count_E_correct += int(pred_E.argmax() == next_E)
                        E_loss += torch.nn.functional.cross_entropy(pred_E, next_E.view(-1))
                        e_div  += 1
                        model.register_predicted_entity(next_E)
                        # model.register_predicted_entity(pred_e.argmax())

                        current_entity = model.get_entity(next_E)
                        
                        pred_L = model.predict_length(h_t, current_entity)
                        L_loss += torch.nn.functional.cross_entropy(pred_L, next_L.view(-1))
                        
                        l_div  += 1
                    else:
                        # no entity
                        if pred_R.argmax():
                            # wrong True pred
                            r_false_positive += 1
                            # extra loss
                            R_loss += r_current_loss * rz_amplifier
                        else:
                            # correct False pred
                            R_loss += r_current_loss
                else:
                    # 2. Otherwise
                    # last L unequal 1, continuing entity mention
                    # setting last new_L = last_L - 1
                    # new_R = last_R
                    # new_E = last_E
                    # not necessary as training data is already taken care of it
                    
                    pred_E = model.predict_entity(h_t, t)
                    count_E += 1
                    count_E_correct += int(pred_E.argmax() == next_E)
                    E_loss += torch.nn.functional.cross_entropy(pred_E, next_E.view(-1))
                    e_div  += 1
                    pass
                
                # 3. Sample X
                pred_X = model.predict_word(h_t, current_entity)
                X_loss += torch.nn.functional.cross_entropy(pred_X, next_X.view(-1))
                # 4. Advance the RNN on predicted token, here in training next token 
                h_t, states = model.forward_rnn(doc.X[t+1])
                h_t = h_t.squeeze(0)
                # new hidden state of next token from here (h_t, previous was actually h_t-1)
                
                # 5. Update
                if next_R == 1:
                    model.update_entity(next_E, h_t, t)
                    current_entity = model.get_entity(next_E)
                    
                    
                # 6. Nothing toDo?
                
            ## End of Paper Algorithm
            
            X_epoch_loss += X_loss.item()
            R_epoch_loss += R_loss.item()
            E_epoch_loss += E_loss.item()
            L_epoch_loss += L_loss.item()
            X_loss = X_loss / len(doc)
            R_loss = R_loss / max(r_div, 1)         
            E_loss = E_loss / max(e_div, 1)
            L_loss = L_loss / max(l_div, 1)
            
            epoch_tokens += len(doc)
            epoch_r_div  += r_div
            epoch_l_div += l_div
            epoch_e_div += e_div

            if optimizer:
                optimizer.zero_grad()
                loss = X_loss + R_loss + E_loss + L_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                r_prec   = r_true_positive / max((r_true_positive+r_false_positive), 1)
                r_recall = r_true_positive / max(count_R, 1)
                print(f'Doc {i_doc}/{len(corpus)-1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct/count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}')
                sys.stdout.flush()
        
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f'{h} hours and '*bool(h)
        if optimizer:
            print(f'Epoch {epoch} finished after {x_hour_and_}{m} minutes.')
        else:
            print(f'Evaluation on "{corpus.partition}" partition finished after {x_hour_and_}{m} minutes.')
        r_prec   = r_true_positive / max((r_true_positive+r_false_positive), 1)
        r_recall = r_true_positive / max(count_R, 1)
        rf_score  = 2*((r_prec*r_recall)/max(r_prec+r_recall, 1))

        print(f'Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct/count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}, R_Fscore {rf_score:0.3}')
        #print(f'GPU Mem: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.max_memory_allocated(0)/1024**3,1)} GB, {round(torch.cuda.memory_cached(0)/1024**3,1)}/{round(torch.cuda.max_memory_cached(0)/1024**3,1)} GB')
        print()

        if optimizer:
            file_name = str_pattern.format(model.__class__.__name__, model.lstm.hidden_size, epoch)
            save_model(model, file_name)
            if eval_corpus:
                with torch.no_grad():
                    model.eval()
                    run_nlm(model, eval_corpus, status_interval=None, rz_amplifier=rz_amplifier)
                    model.train()

def run_lme(model, corpus, optimizer=None, epochs=1, eval_corpus=None, status_interval=25, str_pattern='{}_{}_epoch_{}.pkl', rz_amplifier=10):
    if optimizer:
        print(f'Start training {model.__class__.__name__} with {model.lstm.hidden_size} units for {epochs} epoch(s).')
    for epoch in range(1, epochs+1):
        X_epoch_loss, E_epoch_loss, Z_epoch_loss = 0, 0, 0
        epoch_tokens, epoch_e_div = 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_Z = 0
        z_true_positive  = 0
        z_false_positive = 0
        
        zero_target = torch.zeros(1, dtype=torch.long, device=device)
        for i_doc, doc in enumerate(corpus.gen()):
            model.reset_state()

            doc_num_tokens    = doc.X.size(0)-1
            
            X_loss = torch.tensor(0, dtype=torch.float, device=device)
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            Z_loss = torch.tensor(0, dtype=torch.float, device=device)
            
            e_div = torch.sum(doc.Z).item()
            
            for t in range(doc_num_tokens): 
                x_target = doc.X[t+1].unsqueeze(0)
                e_target = doc.E[t+1].unsqueeze(0)
                z_target = doc.Z[t+1]
                x_out, z_i, p_v = model.train_forward(doc.X[t], e_target)
                if z_i > 0.5:
                    if z_target:
                        z_true_positive  += 1
                    else:
                        z_false_positive += 1
            
                # token loss
                X_loss += F.cross_entropy(x_out, x_target)
                # z_i loss
                Z_loss += F.binary_cross_entropy(z_i.squeeze(), z_target)*rz_amplifier
                
                # entity loss
                if e_target:
                    # adding new entity not occurred before
                    if p_v.size(0) == e_target:
                        E_loss += F.cross_entropy(p_v.unsqueeze(0), zero_target)
                        count_E_correct += int(p_v.argmax() == 0)
                    else:
                        E_loss += F.cross_entropy(p_v.unsqueeze(0), e_target)
                        count_E_correct += int(p_v.argmax() == e_target)
                    
                    count_E += 1
                     
            X_epoch_loss += X_loss.item()
            E_epoch_loss += E_loss.item()
            Z_epoch_loss += Z_loss.item()
            X_loss = X_loss / doc_num_tokens
            E_loss = E_loss / max(e_div, 1)
            Z_loss = Z_loss / doc_num_tokens       
            
            
            epoch_tokens += doc_num_tokens
            epoch_e_div += e_div
            count_Z += torch.sum(doc.Z).item()

            if optimizer:
                optimizer.zero_grad()
                loss = X_loss + E_loss + Z_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                z_prec   = z_true_positive / max((z_true_positive+z_false_positive), 1)
                z_recall = z_true_positive / max(count_Z, 1)
                print(f'Doc {i_doc}/{len(corpus)-1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, Z_loss {Z_epoch_loss / epoch_tokens:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, E_acc {count_E_correct/count_E:0.3}, Z_prec {z_prec:0.3}, Z_recall {z_recall:0.3}')
                sys.stdout.flush()
        
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f'{h} hours and '*bool(h)
        if optimizer:
            print(f'Epoch {epoch} finished after {x_hour_and_}{m} minutes.')
        else:
            print(f'Evaluation on "{corpus.partition}" partition finished after {x_hour_and_}{m} minutes.')
        z_prec   = z_true_positive / max((z_true_positive+z_false_positive), 1)
        z_recall = z_true_positive / max(count_Z, 1)
        zf_score  = 2*((z_prec*z_recall)/max(z_prec+z_recall, 1))

        print(f'Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, Z_loss {Z_epoch_loss / epoch_tokens:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, E_acc {count_E_correct/count_E:0.3}, Z_prec {z_prec:0.3}, Z_recall {z_recall:0.3}, Z_Fscore {zf_score:0.3}')
        #print(f'GPU Mem: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.max_memory_allocated(0)/1024**3,1)} GB, {round(torch.cuda.memory_cached(0)/1024**3,1)}/{round(torch.cuda.max_memory_cached(0)/1024**3,1)} GB')
        print()

        if optimizer:
            file_name = str_pattern.format(model.__class__.__name__, model.lstm.hidden_size, epoch)
            save_model(model, file_name)
            if eval_corpus:
                with torch.no_grad():
                    model.eval()
                    run_lme(model, eval_corpus, status_interval=None, rz_amplifier=rz_amplifier)
                    model.train()
                    
                    
