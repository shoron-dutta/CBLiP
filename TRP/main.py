from preprocessing import TransductiveData
from torch.utils.data import DataLoader
from model import CBModel
from datetime import timedelta
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
from torch import nn
import  copy
import torch, time, json
from parser_ import create_parser
# torch.manual_seed(12)


train_mrr = []
eval_mrr = []

train_mr = []
eval_mr = []

train_hits_1 = []
eval_hits_1 = []

train_hits_3 = []
eval_hits_3 = []

train_acc = []
eval_acc = []

train_losses = []
eval_losses = []


def train(model, train_data):
    model.train()  # turn on train mode
    total_loss = 0.
    epoch_sum_ranks, epoch_sum_rr, epoch_sum_hits_1, epoch_sum_hits_3 = 0., 0., 0., 0.
    for _, sample in enumerate(train_data):
        optimizer.zero_grad()
        
        output = model(sample['target_triple'].to(device), sample['h_neighbors'].to(device), \
                sample['t_neighbors'].to(device), sample['n'].to(device), sample['adj'].to(device))
        loss = criterion(output, sample['target_triple'][:,1].to(device))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, idx = output.sort(dim=1, descending=True)
        ranks = (sample['target_triple'][:,1].to(device).unsqueeze(1)==idx).nonzero()[:,1] + 1
        
        epoch_sum_ranks += ranks.sum().item()
        epoch_sum_rr += (1/ranks).sum().item()
        epoch_sum_hits_1 += (ranks==1).sum().item()
        epoch_sum_hits_3 += (ranks <= 3).sum().item()
    train_mr.append(epoch_sum_ranks/len(train_data.dataset))
    train_mrr.append(epoch_sum_rr/len(train_data.dataset))
    train_hits_1.append(epoch_sum_hits_1/len(train_data.dataset)) 
    train_hits_3.append(epoch_sum_hits_3/len(train_data.dataset))  
    train_losses.append(total_loss / len(train_data))
    
    return total_loss


def evaluate(model, eval_data):
    model.eval()  # turn on evaluation mode

    total_loss = 0.
    epoch_sum_ranks, epoch_sum_rr, epoch_sum_hits_1, epoch_sum_hits_3 = 0., 0., 0., 0.
    with torch.no_grad():
        for _, sample in enumerate(eval_data):
            output = model(sample['target_triple'].to(device), sample['h_neighbors'].to(device), \
                sample['t_neighbors'].to(device), sample['n'].to(device), sample['adj'].to(device))
            
            total_loss += criterion(output, sample['target_triple'][:,1].to(device)).item()
            
            _, idx = output.sort(dim=1, descending=True)
            ranks = (sample['target_triple'][:,1].to(device).unsqueeze(1)==idx).nonzero()[:,1] + 1
            
            epoch_sum_ranks += ranks.sum().item()
            epoch_sum_rr += (1/ranks).sum().item()
            epoch_sum_hits_1 += (ranks==1).sum().item()
            epoch_sum_hits_3 += (ranks <= 3).sum().item()

    eval_mr.append(epoch_sum_ranks/len(eval_data.dataset))
    eval_mrr.append(epoch_sum_rr/len(eval_data.dataset))
    eval_hits_1.append(epoch_sum_hits_1/len(eval_data.dataset)) 
    eval_hits_3.append(epoch_sum_hits_3/len(eval_data.dataset))  
    eval_losses.append(total_loss / len(eval_data))
    return  total_loss / len(eval_data), \
            epoch_sum_rr/len(eval_data.dataset), \
            epoch_sum_hits_1/len(eval_data.dataset)
    

def evaluate_test(model, test_data, epoch_):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    epoch_sum_ranks, epoch_sum_rr, epoch_sum_hits_1, epoch_sum_hits_3 = 0., 0., 0., 0.
    with torch.no_grad():
        for _, sample in enumerate(test_data):
            output = model(sample['target_triple'].to(device), sample['h_neighbors'].to(device), \
                    sample['t_neighbors'].to(device), sample['n'].to(device), sample['adj'].to(device))
            
            total_loss += criterion(output, sample['target_triple'][:,1].to(device)).item()
            _, idx = output.sort(dim=1, descending=True)
            ranks = (sample['target_triple'][:,1].to(device).unsqueeze(1)==idx).nonzero()[:,1] + 1
            
            epoch_sum_ranks += ranks.sum().item()
            epoch_sum_rr += (1/ranks).sum().item()
            epoch_sum_hits_1 += (ranks==1).sum().item()
            epoch_sum_hits_3 += (ranks <= 3).sum().item()

    
    metrics_dict = {}
    metrics_dict['test_mr'] = round(epoch_sum_ranks/len(test_data.dataset), 3)
    metrics_dict['test_mrr'] = round(epoch_sum_rr/len(test_data.dataset), 3)
    metrics_dict['test_hits_1'] = round(epoch_sum_hits_1/len(test_data.dataset), 3)
    metrics_dict['test_hits_3'] = round(epoch_sum_hits_3/len(test_data.dataset), 3)
    metrics_dict['test_loss'] = round(total_loss / len(test_data), 3)
    metrics_dict['best_epoch'] = epoch_

    return metrics_dict


if __name__ == '__main__':

    st = time.time()    
    args, suffix, device, filepath = create_parser('transductive')
    
    data = TransductiveData(args, 2) # use base = 2 for distance information
    model = CBModel(args, device, data.e_idx, data.r_idx)
    model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'pytorch_total_params: {pytorch_total_params}')

    
    best_valid_loss = float('inf')
    best_valid_mrr = 0.
    best_valid_hits_1 = 0.
    
    best_model_loss = None
    best_model_mrr = None
    best_model_hits_1 = None
    
    best_epoch_loss = 0
    best_epoch_mrr = 0
    best_epoch_hits_1 = 0

    data_iterator  = DataLoader(data, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=True) 
    for epoch in tqdm(range(args.num_epochs)):
        
        data.shuffle_neighborhood()
        data.set_mode('train')
        train_loss = train(model, data_iterator)
        data.set_mode('valid')
        valid_loss, valid_mrr, valid_hits_1 = evaluate(model, data_iterator)
        
        if valid_hits_1 > best_valid_hits_1:
            best_valid_hits_1 = valid_hits_1
            best_model_hits_1 = copy.deepcopy(model)
            best_epoch_hits_1 = epoch
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            best_model_mrr = copy.deepcopy(model)
            best_epoch_mrr = epoch
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_loss = copy.deepcopy(model)
            best_epoch_loss = epoch

        # test model and plot losses after every LOG epochs
        if (epoch % args.log == 0 and epoch>0) or epoch == args.num_epochs-1:
            print('Best epoch_loss:\t', best_epoch_loss)
            print('Best epoch_mrr:\t\t', best_epoch_mrr)
            print('Best epoch_hits_1:\t', best_epoch_hits_1)
            
            test_metrics = dict()
            data.set_mode('test')
            test_metrics["loss"] = evaluate_test(best_model_loss, data_iterator, best_epoch_loss)
            test_metrics["mrr"] = evaluate_test(best_model_mrr, data_iterator, best_epoch_mrr)
            test_metrics["hits_1"] = evaluate_test(best_model_hits_1, data_iterator, best_epoch_hits_1)
            execution_time = time.time()-st
            test_metrics["current_epoch"] = epoch
            test_metrics["time"] = str(timedelta(seconds=execution_time))

            pp.pprint(test_metrics)
            
            with open(filepath + '_test_results_' + suffix +'.json', 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=4)
            
            
            torch.save(best_model_loss.state_dict(), filepath + suffix +'_model.pt')

    print('training done!')
    print('Metrics\t\t\tloss\tMRR\tHits_1')
    print(f'Best epoch:\t\t {best_epoch_loss}\t{best_epoch_mrr}\t{best_epoch_hits_1}')
    print(f'Training hits@1:\t {train_hits_1[best_epoch_loss]:.3f}\t{train_hits_1[best_epoch_mrr]:.3f}\t{train_hits_1[best_epoch_hits_1]:.3f}')
    print(f'Training hits@3:\t {train_hits_3[best_epoch_loss]:.3f}\t{train_hits_3[best_epoch_mrr]:.3f}\t{train_hits_3[best_epoch_hits_1]:.3f}')
    print(f'Training MR:\t\t {train_mr[best_epoch_loss]:.3f}\t{train_mr[best_epoch_mrr]:.3f}\t{train_mr[best_epoch_hits_1]:.3f}')
    print(f'Training MRR:\t\t {train_mrr[best_epoch_loss]:.3f}\t{train_mrr[best_epoch_mrr]:.3f}\t{train_mrr[best_epoch_hits_1]:.3f}')
    print(f'Training loss:\t\t {train_losses[best_epoch_loss]:.3f}\t{train_losses[best_epoch_mrr]:.3f}\t{train_losses[best_epoch_hits_1]:.3f}\n')

    print(f'Validation hits@1:\t {eval_hits_1[best_epoch_loss]:.3f}\t{eval_hits_1[best_epoch_mrr]:.3f}\t{eval_hits_1[best_epoch_hits_1]:.3f}')
    print(f'Validation hits@3:\t {eval_hits_3[best_epoch_loss]:.3f}\t{eval_hits_3[best_epoch_mrr]:.3f}\t{eval_hits_3[best_epoch_hits_1]:.3f}')
    print(f'Validation MR:\t\t {eval_mr[best_epoch_loss]:.3f}\t{eval_mr[best_epoch_mrr]:.3f}\t{eval_mr[best_epoch_hits_1]:.3f}')
    print(f'Validation MRR:\t\t {eval_mrr[best_epoch_loss]:.3f}\t{eval_mrr[best_epoch_mrr]:.3f}\t{eval_mrr[best_epoch_hits_1]:.3f}')
    print(f'Validation loss:\t {eval_losses[best_epoch_loss]:.3f}\t{eval_losses[best_epoch_mrr]:.3f}\t{eval_losses[best_epoch_hits_1]:.3f}')
    
    print(f'Completed running exp: {filepath}{suffix}')
    
    