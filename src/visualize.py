import matplotlib.pyplot as plt
import numpy as np
import torch

def smoothen(l, w_l, w_s):
    m = []
    for i in range(0, len(l)-w_l, w_s):
        e = 0
        for j in range(i, i+w_l):
            e += l[j]
        e = e / w_l 
        m.append(e)
    return m

def plot_metric(metric:list, label, window_len:int=100, window_step=1, c:str='b', save:bool=False):
    
    metric = smoothen(metric, window_len, window_step)

    x = np.linspace(1, len(metric), len(metric))
    plt.figure(figsize=(10, 6))
    plt.plot(x, metric, label=label, color=c)
    plt.legend()
    if save == True:
        plt.savefig('metrics_plot')
    plt.show()

def load_metrics(model_checkpoint):
    metrics = torch.load(model_checkpoint)['metrics']
    return metrics


if __name__ == '__main__':
    model_checkpoint = '../models/roberta_3e_e-5.pt'
    base = model_checkpoint.split('/')[-1].split('.')[0]
    metrics = load_metrics(model_checkpoint)
    with open(f'../results/{base}-metrics.txt', 'x') as f:
        f.write('train_losses\n')
        f.write(f'{max(metrics["train_losses"])}\n')
        f.write(f'{np.mean(metrics["train_losses"])}\n')
        f.write(f'{min(metrics["train_losses"])}\n')
        f.write('\n')
        f.write('val_losses\n')
        f.write(f'{max(metrics["val_losses"])}\n')
        f.write(f'{np.mean(metrics["val_losses"])}\n')
        f.write(f'{min(metrics["val_losses"])}\n')
        f.write('\n')
        f.write('train_accs\n')
        f.write(f'{max(metrics["train_accs"])}\n')
        f.write(f'{np.mean(metrics["train_accs"])}\n')
        f.write(f'{min(metrics["train_accs"])}\n')
        f.write('\n')
        f.write('val_accs\n')
        f.write(f'{max(metrics["val_accs"])}\n')
        f.write(f'{np.mean(metrics["val_accs"])}\n')
        f.write(f'{min(metrics["val_accs"])}\n')
        f.write('\n')

    print(f'=> train_losses MAX: {max(metrics["train_losses"])}')
    print(f'=> train_losses MEAN: {np.mean(metrics["train_losses"])}')
    print(f'=> train_losses MIN: {min(metrics["train_losses"])}')

    print(f'=> val_losses MAX: {max(metrics["val_losses"])}')
    print(f'=> val_losses MEAN: {np.mean(metrics["val_losses"])}')
    print(f'=> val_losses MIN: {min(metrics["val_losses"])}')

    print(f'=> train_accs MAX: {max(metrics["train_accs"])}')
    print(f'=> train_accs MEAN: {np.mean(metrics["train_accs"])}')
    print(f'=> train_accs MIN: {min(metrics["train_accs"])}')

    print(f'=> val_accs MAX: {max(metrics["val_accs"])}')
    print(f'=> val_accs MEAN: {np.mean(metrics["val_accs"])}')
    print(f'=> val_accs MIN: {min(metrics["val_accs"])}')

    plot_metric(metrics['train_losses'], 'train_losses', c='orange')
    plot_metric(metrics['val_losses'], 'val_losses', c='orange')
    plot_metric(metrics['train_accs'], 'train_accs', c='blue')
    plot_metric(metrics['val_accs'], 'val_accs', c='blue')




