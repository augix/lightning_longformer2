import torch
from torch.utils.data import Dataset

# required config attributes:
# config.n_input_values
# config.seq_len
# config.mask_frac
# config.mask_token
# config.batch_size

# --------------------------------
#  Create data for testing
# --------------------------------
def make_seq(n_input_values, seq_len):
    seq = torch.randint(low=1, high=n_input_values, size=(1, seq_len)) # leave 0 for mask
    # make 2nd half of sequence equal to 1st half
    half_len = seq_len // 2
    seq[:, half_len:] = seq[:, :half_len].flip(1)
    return seq

def masking(seq, mask_frac):
    seq_len = seq.size(1)
    mask = torch.rand(size=(1, seq_len)) < mask_frac
    masked_seq = torch.where(mask, torch.zeros_like(seq), seq)
    return masked_seq, mask

def make_record(config):
    seq = make_seq(config.n_input_values, config.seq_len)
    ids = torch.arange(config.seq_len).reshape(1, -1)
    masked_seq, mask = masking(seq, config.mask_frac)
    return {'pos_id': ids, 'input': masked_seq, 'target': seq, 'mask': mask}

def make_batch(config):
    records = [make_record(config) for _ in range(config.batch_size)]
    batch = {
        'pos_id': torch.cat([record['pos_id'] for record in records]),
        'input': torch.cat([record['input'] for record in records]),
        'target': torch.cat([record['target'] for record in records]),
        'mask': torch.cat([record['mask'] for record in records])}
    return batch

class theDataset(Dataset):
    def __init__(self, config, sample_size):
        self.config = config
        self.size = sample_size

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        record = make_record(self.config)
        return {
            'pos_id': record['pos_id'].squeeze(0),
            'input': record['input'].squeeze(0),
            'target': record['target'].squeeze(0),
            'mask': record['mask'].squeeze(0),
        }

