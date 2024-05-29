import torch
from torch.nn.functional import softmax
from torch.nn import DataParallel
from model import BERTModel
from transformers import AutoTokenizer
from torchsummary import summary

if __name__ == '__main__':

    text = {"evidence": "ChuChu TV is the 42nd most subscribed YouTube channel in the world , with more than 26 million subscribers across 8 channels .", 
            "claim": "The YouTube channel Chuchu TV is placed 42nd and has more than 25 million subscribers ."}
    
    model_checkpoint = '../models/roberta_3e_e-5.pt'
    state_dict = torch.load(model_checkpoint)['state_dict']

    model = model = DataParallel(BERTModel('roberta'))
    model.load_state_dict(state_dict)
    print(model)
#     model.eval()
    
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#     tokens = tokenizer(text['evidence'], text['claim'], padding='max_length', max_length=512, truncation='only_first')

#     input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
#     attention_mask =  torch.tensor(tokens['attention_mask']).unsqueeze(0)
#     token_type_ids =  torch.tensor(tokens['token_type_ids']).unsqueeze(0)
#     out = model(input_ids, attention_mask, token_type_ids)
#     out = softmax(out).squeeze()

#     print(f'=> Supports[%]: {out[0]}')
#     print(f'=> Refutes[%]: {out[2]}')
#     print(f'=> NotEnoughInfo[%]: {out[1]}')

    
    