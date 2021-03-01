
from models import LMEntity, LMEntity_mod
from dataset import MyDataset
from helpers import file2model

import torch
import torch.nn as nn
import torch.utils.data as data

import time
import sys
import os
import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, model_file=None, verbose=False, stop_point=None, data_dir='data', eval_file='__eval.dict', eval_path='.'):
    model.eval()
    start_eval = time.time()
    # creating Dataset with dev data
    dev_dataset = MyDataset(data_dir=data_dir, partition='dev', verbose=False)
    # fetching vocabulary
    # vocab = dev_dataset.vocab 

    # creating DataLoader
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 6}

    gen = data.DataLoader(dev_dataset, **params)

    # defining loss functions
    # loss function for tokens and entities
    xe_crit = nn.CrossEntropyLoss()
    # loss function for z_i 
    z_crit = torch.nn.BCELoss()

    # Initialize loss tensors
    x_loss = torch.tensor(0, dtype=torch.float, device=device)
    z_loss = torch.tensor(0, dtype=torch.float, device=device)
    e_loss = torch.tensor(0, dtype=torch.float, device=device)
    xe_loss = torch.tensor(0, dtype=torch.float, device=device)
    #loss = torch.tensor(0, dtype=torch.float, device=device)

    loss_data = []
    
    num_docs = len(dev_dataset)
    token_num_all = 0
    z_sum_all = 0
    with torch.no_grad():
        current_doc = 0
        for X, y, z in gen:      
            # moving data to device
            X = X.to(device)
            y = y.to(device)
            z = z.to(device)
            doc_num_tokens = X.size(1) - 2
            for i in range(doc_num_tokens):
                x_target = X[0][i+1].unsqueeze(0)
                e_target = y[0][i+1].unsqueeze(0)
                z_target = z[0][i+1]
                x_out, z_i, p_v = model.train_forward(X[0][i], e_target)
                
                # token loss
                current_xe_loss = xe_crit(x_out, x_target)
                x_loss += current_xe_loss
                # z_i loss
                z_loss += z_crit(z_i.squeeze(), z_target)
                # entity loss
                if e_target:
                    # adding entity token loss
                    xe_loss += current_xe_loss
                    
                    # adding new entity not occurred before
                    if p_v.size(0) == e_target:
                        e_loss += xe_crit(p_v.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
                    else:
                        e_loss += xe_crit(p_v.unsqueeze(0), e_target)
                        
            
            current_loss =  {'x_loss': x_loss.item(),
                             'z_loss': z_loss.item(),
                             'e_loss': e_loss.item(),
                             'xe_loss': xe_loss.item(),
                             'loss': (x_loss + z_loss + e_loss).item(),
                             'doc_num_tokens': doc_num_tokens,
                             'sum_z': torch.sum(z).item()}
            token_num_all += doc_num_tokens
            z_sum_all += torch.sum(z).item()
            

            loss_data.append(current_loss)
            if verbose:
                print(f'Document: {current_doc} ({doc_num_tokens} tokens)')
                print(current_loss)
                
            x_loss = torch.tensor(0, dtype=torch.float, device=device)
            z_loss = torch.tensor(0, dtype=torch.float, device=device)
            e_loss = torch.tensor(0, dtype=torch.float, device=device)
            xe_loss = torch.tensor(0, dtype=torch.float, device=device)
            
            current_doc += 1    
            model.reset_state()
            if stop_point and current_doc >= stop_point:
                break
    print(f'Evaluation finished in {time.time()-start_eval} sec.')
    full_x_loss = round(sum(map(lambda loss_dict: loss_dict['x_loss'], loss_data))/token_num_all, 2)
    full_z_loss = round(sum(map(lambda loss_dict: loss_dict['z_loss'], loss_data))/token_num_all, 2)
    full_e_loss = round(sum(map(lambda loss_dict: loss_dict['e_loss'], loss_data))/z_sum_all, 2)
    full_xe_loss = round(sum(map(lambda loss_dict: loss_dict['xe_loss'], loss_data))/z_sum_all, 2)
    full_loss = round(sum(map(lambda loss_dict: loss_dict['loss'], loss_data))/token_num_all, 2)
    full_loss_dict = {'full_x_loss': full_x_loss,
                      'full_z_loss': full_z_loss,
                      'full_e_loss': full_e_loss,
                      'full_xe_loss': full_xe_loss,
                      'full_loss': full_loss}
    if model_file:
        eval_dict = dict()
        eval_file_path = os.path.join(eval_path, eval_file)
        if os.path.isfile(eval_file_path):
            with open(eval_file) as f:
                eval_dict = eval(f.read())
        eval_dict[model_file] = full_loss_dict
        with open(eval_file_path, 'w') as f:
            f.write(repr(eval_dict))
            print(f"Evaluation dictionary '{eval_file_path}' updated for '{model_file}'.")

    del dev_dataset, gen, X, y, z
    return loss_data, full_loss_dict


if __name__ == '__main__':

    __eval_dict = '__eval.dict'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str,
                    help="path to epoch save file")

    parser.add_argument("-p", "--path", type=str, default='.',
                    help="path directory containing model files")

    parser.add_argument("-d", "--dict", action="store_true",
                        help=f"outputs the results stored in '{__eval_dict}' instead of calculating from model files")

    parser.add_argument("-a", "--all", action="store_true",
                        help="evaluates all save files in the current directory")

    parser.add_argument("--force", action="store_true",
                        help=f"if set evaluation is forced to recalculate all model files including those already in '{__eval_dict}'")

    parser.add_argument("-n", "--top", type=int, default=3,
                        help="when flag -a is set, shows only the top n models")
    parser.add_argument("-m", "--mod", action="store_true",
						help="if flag set modded class will be used")



    args = parser.parse_args()
    if args.dict:
        file_path = os.path.join(args.path, __eval_dict)
        with open(file_path) as f:
            eval_dict = eval(f.read())
        eval_list = eval_dict.items()
        top_n = list(sorted(eval_list, key=lambda x: x[1]['full_x_loss']))[:args.top]
        pprint.pprint(top_n)
    elif args.all:
        eval_list = []
        dict_path = file_path = os.path.join(args.path, __eval_dict)
        eval_dict = None
        if os.path.isfile(dict_path):
            with open(dict_path) as f:
                eval_dict = eval(f.read())
        files = list(sorted(filter(lambda x: x.endswith('.pkl'), os.listdir(args.path))))

        for file in files:
            if eval_dict and file in eval_dict:
                print(f"Model file '{file}' already evaluated in '{dict_path}':")
                if args.force:
                    print(f" - Overwriting existing evaluation for '{file}'\n")
                else:
                    print(f" - Omitting '{file}'\n")
                    continue
            file_path = os.path.join(args.path, file)
            model = file2model(file_path, args.mod)
            loss_data, full_loss_dict = evaluate_model(model, model_file=file, eval_path=args.path)
            eval_list.append((file, full_loss_dict))
            sys.stdout.flush()
        top_n = list(sorted(eval_list, key=lambda x: x[1]['full_x_loss']))[:args.top]
        pprint.pprint(top_n)
    elif args.file:
        model = file2model(args.file, args.mod)
        print(f"Start evaluating '{args.file}'")
        loss_data, full_loss_dict = evaluate_model(model, model_file=args.file, eval_path=args.path)
        import pprint
        pprint.pprint(full_loss_dict)
    print('Evaluation finished.')


