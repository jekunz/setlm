import torch
import os
from torch.utils import data
from random import shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir='data', partition='train', extend_vocab=True, get_lengths=False, ID_store={}, vocab={}, gvocab={}, auto_load = True, verbose = True):
        #super().__init__()
        'Initialization'
        self.ID_store = ID_store
        self.data_dir = data_dir
        self.partition = partition
        self._vocab = vocab
        self._gvocab = gvocab
        self.extend_vocab = extend_vocab
        self.get_lengths = get_lengths
        self.verbose = verbose
        if auto_load:
            self.load_file_IDs()
    
    def load_file_IDs(self, id_file='file_ids.dict'):
        with open(f'{self.data_dir}/{id_file}') as f:
            self.ID_store = eval(f.read())
        if self.verbose:
            print(f"{len(self)} file IDs (documents) for partition '{self.partition}' succesfully loaded.")

        with open(f'{self.data_dir}/vocab.dict') as f:
            self._vocab = eval(f.read())
        if self.verbose:
            print(f"Vocab dict with vocabulary of {len(self._vocab)} tokens loaded.")
        if os.path.isfile(f'{self.data_dir}/gvocab.dict') and self.extend_vocab:
            with open(f'{self.data_dir}/gvocab.dict') as f:
                self._gvocab = eval(f.read())
            if self.verbose:
                print(f"Extended gvocab dict with vocabulary of {len(self._gvocab)} tokens loaded.")

    @property
    def vocab(self):
        return max((self._vocab, self._gvocab), key=len)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID_store[self.partition])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.ID_store[self.partition][index]

        # Load data and get label
        X = torch.load(f'{self.data_dir}/{self.partition}/doc_{ID:03}_tokens.pt') #map_location=lambda storage, loc: storage.cuda(0))
        y = torch.load(f'{self.data_dir}/{self.partition}/doc_{ID:03}_entities.pt')
        #map_location=lambda storage, loc: storage.cuda(0))
        z = torch.zeros(y.size(0), dtype=torch.float)
        z[y > 0] = 1
        z = z.contiguous()
        if self.get_lengths:
            l = torch.load(f'{self.data_dir}/{self.partition}/doc_{ID:03}_lengths.pt')
            return X, y, z, l
        return X, y, z


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class SharedFunc:
    def calcIndices(self, signal_token):
        tokens = self.data['tokens']
        stoken_indices = (tokens == signal_token).nonzero().view(-1).tolist()
        stoken_indices[0] = 0 #change start of first sent due to doc_start
        stoken_indices.append(tokens.size(0)) # adding end token
        indices = []
        for i in range(1, len(stoken_indices)):
            start = stoken_indices[i-1]; end = stoken_indices[i];
            indices.append((start, end))
        return indices
    
    def to_(self, device):
        for key in self.data:
            self.data[key] = self.data[key].to(device)
    
    @property 
    def X(self):
        return self.data['tokens']
    @property
    def E(self):
        return self.data['entities']
    @property
    def R(self):
        if isinstance(self, CorpusLoader): raise AttributeError('Function not Available for "CorpusLoader"')
        return self.data['is_entity']
    @property
    def Z(self):
        return self.R
    @property
    def L(self):
        return self.data['lengths']


class SentEmbedding(SharedFunc):
    def __init__(self, data, index=None):
        self.data = data
        self.index = index
        
    def __len__(self):
        return len(self.X.size(0))
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
    
    def __getitem__(self, key):
        return self.data[key]
    

class DocLoader(SharedFunc):
    def __init__(self, data_dict, start_sent, index=None, corpus=None, len_mode='token'):
        assert len_mode in ('token', 'sent')
        self.len_mode = len_mode
        self.start_sent = start_sent
        self.index = index
        self.data = data_dict
        self.corpus = corpus
        self.indices = self.calcIndices(start_sent)
        self.calcEntityBool()
        assert self.X.size(0) == self.R.size(0) == self.E.size(0)


    
    def calcEntityBool(self):
        self.data['is_entity'] = torch.zeros(self.data['tokens'].size(0), dtype=self.corpus.rz_dtype, device=device)#.to(device)
        #self.data['is_entity'] = torch.LongTensor(self.data['tokens'].size(0)).zero_()#.to(device)
        self.data['is_entity'][self.data['entities'] > 0] = 1
        
    def __len__(self):
        if self.len_mode == 'sent':
            return len(self.indices)
        elif self.len_mode == 'token':
            return self.X.size(0)
    
    def __getitem__(self, index):
        sent_data = dict(map(lambda k: (k, self.data[k][slice(*self.indices[index])]), self.data))
        return SentEmbedding(sent_data, index=index)
    
    def __str__(self):
        s = f"<Doc id {self.index}: {len(self)} Sentences and {self.data['tokens'].size(0)} tokens>\n"
        for i, sent in enumerate(self):
            s += f"{i}:" + ' '.join(map(self.corpus.id2token, sent['tokens'].tolist())) + '\n' # toDo add UNK tokens
        return s

    def __repr__(self):
        return f"<Doc {self.index}: {len(self)} Sentences and {self.data['tokens'].size(0)} Tokens>\n{self.data['tokens']}"
    

class CorpusLoader(SharedFunc):
    def __init__(self, data_dir='data', partition='train', str_pattern='{}_{}.pt', start_sent=4, start_doc=2, lengths=True, l_min=1, return_Tensor=False, set_shift=False, entity_solo=False):
        self.start_doc  = start_doc
        self.start_sent = start_sent
        self.unk_token  = 1
        self.embedding_shift = set_shift
        self.l_min    = l_min
        self.entity_solo = entity_solo
        if lengths:
            self.rz_dtype = torch.long
        else:
            self.rz_dtype = torch.float
        self.data_dir = data_dir
        self.str_pattern = str_pattern
        self.return_Tensor = return_Tensor
        self.partition  = partition
        self.file_types = ['tokens', 'entities']
        if lengths:
            self.file_types.append('lengths')
        self.data = {}
        self.load()
        self.indices = self.calcIndices(self.start_doc)
        self.to_(device)
        print(f"{len(self.indices)} documents found.")
        
    def getPath(self, file_type):
        return os.path.join(self.data_dir, self.partition, f'{file_type}.pt')
        
    def load(self):
        print(f"Loading partition: '{self.partition}' ..")
        for ft in self.file_types:
            try:
                self.data[ft] = torch.load(self.getPath(ft)).contiguous()
            except FileNotFoundError as e:
                print(f"Could not find data for partition '{self.partition}'.")
                raise e
        with open(os.path.join(self.data_dir, 'vocab.dict')) as f:
            self.vocab = eval(f.read())
            self._id2token = dict(map(reversed, self.vocab.items()))
        with open(os.path.join(self.data_dir, 'file_ids.dict')) as f:
            self.file_ids = eval(f.read())
            
        if 'lengths' in self.data:
            current_l_min = self.data['lengths'].min().item()
            if current_l_min != self.l_min:
                difference = self.l_min - current_l_min
                self.data['lengths'] = self.data['lengths'] + difference
        if self.embedding_shift and not (0 in self.data['tokens']) :
            self.embedding_shift = 1
            self.data['tokens'] = self.data['tokens'] - self.embedding_shift
            self.start_sent -= self.embedding_shift
            self.start_doc  -= self.embedding_shift
            self.unk_token  -= self.embedding_shift
    def id2token(self, token_id):
        return self._id2token.get(token_id, self.unk_token)
    
    def check_entities(self):
        for i in range(len(self)):
            doc_data = dict(map(lambda k: (k, self.data[k][slice(*self.indices[i])]), self.data))
            max_e = 0
            for e in doc_data['entities'].tolist():
                assert e <= max_e + 1, f'doc_id {self.file_ids[self.partition][i]}: {e} <= {max_e} + 1'
                max_e = max(e, max_e)
        print(f'{i+1} documents checked, all entity ids ok!') 
    
    def getByID(self, doc_id):
        return self[self.file_ids[self.partition].index(doc_id)]
    
    def gen(self, shuffle_docs=True):
        index_values = list(range(len(self)))
        if shuffle_docs:
            shuffle(index_values)
        for i in index_values:
            yield self[i]
    
    @property
    def vocab_size(self):
        return len(self.vocab)+1
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        doc_data = dict(map(lambda k: (k, self.data[k][slice(*self.indices[index])]), self.data))
        doc = DocLoader(doc_data, self.start_sent, index=self.file_ids[self.partition][index], corpus=self)
        if not self.return_Tensor:
            return doc
        else:
            if len(doc_data) == 3:
                return doc.X, doc.E, doc.Z
            else:
                return doc.X, doc.E, doc.R, doc.L
            