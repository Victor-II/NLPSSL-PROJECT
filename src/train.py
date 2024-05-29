import torch
import math
import os
import gc
from model import BERTModel
from data import DS, load_combined_dataset, DS_PATH
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, DataParallel
from tqdm import tqdm


SAVE_BEST = True
MODEL = 'roberta'
BATCH_SIZE = 28
LR = 0.00001
EPOCHS = 3
MODEL_SAVE_PATH = f'../models/{MODEL}_{EPOCHS}e_e-5.pt'

def accuracy(out, label):
    _, max_idx_class = out.max(dim=1)
    n = out.size(0)
    assert( n == max_idx_class.size(0))
    acc = (max_idx_class == label).sum().item() / n

    return acc

def train():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # build model
    model = DataParallel(BERTModel(MODEL))
    model.to(device)
    # build datasets
    train_data = load_combined_dataset('train')
    val_data = load_combined_dataset('validation')
    print()
    print('_'*50)
    print('using the following datasets: ', [ds for ds in DS_PATH.keys()])
    print(f'{len(train_data) = }')
    print(f'{len(val_data) = }')
    print('_'*50)
    print()
    train_ds = DS(train_data, MODEL)
    val_ds = DS(val_data, MODEL)
    # build dataloaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    # build optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    # build loss function
    criterion = CrossEntropyLoss()
    
    metrics = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    for epoch in range(EPOCHS):
        # train step
        model.train()
        batch_iterator = tqdm(train_dl, desc=f'Processing epoch {epoch+1:02d}')
        prev_loss = 100

        for batch in batch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch.keys() else None
            label = batch['label'].to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, label)
            acc = accuracy(out.cpu().detach(), label.cpu().detach())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            metrics['train_losses'].append(loss.cpu().detach().numpy())
            metrics['train_accs'].append(acc)

            # os.system('clear')
            # print(torch.cuda.memory_summary(device='cuda:1', abbreviated=True))
            # print(torch.cuda.memory_summary(device='cuda:0', abbreviated=True))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}', 'acc': f'{acc:6.3f}'})
            
        # evaluation step
        model.eval()
        batch_iterator = tqdm(val_dl, desc=f'Evaluating epoch {epoch+1:02d}')
        for batch in batch_iterator:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch.keys() else None
                label = batch['label'].to(device)

                out = model(input_ids, attention_mask, token_type_ids)
                val_loss = criterion(out, label)
                val_acc = accuracy(out.cpu().detach(), label.cpu().detach())
            
            metrics['val_losses'].append(val_loss.cpu().detach().numpy())
            metrics['val_accs'].append(val_acc)
            # print('=> memory_allocated: ' + str(torch.cuda.memory_allocated(0) // 1024 ** 2))
            
            batch_iterator.set_postfix({'val_loss': f'{val_loss.item():6.3f}', 'val_acc': f'{val_acc:6.3f}'})
            
        try:
            os.remove(MODEL_SAVE_PATH)
        except:
            pass
        torch.save({'state_dict': model.state_dict(), 'metrics': metrics}, MODEL_SAVE_PATH)
        print(f'State dict saved at "{MODEL_SAVE_PATH}"')

if __name__ == '__main__':

    train()
    