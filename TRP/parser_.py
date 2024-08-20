import argparse, os
from datetime import datetime, date, timedelta
from os import listdir
from os.path import isfile, join
def process_args(args, dir_name):
    if args.cpu:
        device = 'cpu'
    else:
        device='cuda'  
    filepath = './' + dir_name + '/' + args.path + '/'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    
    # find existing args files and increment by 1 to assign new suffix values
    suffix_list = [int(f[5:-4]) for f in listdir(filepath) if isfile(join(filepath, f)) and f.startswith('args_')]
    suffix = max(suffix_list) + 1 if len(suffix_list)>0 else len(suffix_list) + 1
    suffix = str(suffix)

    print(date.today().strftime("%B %d, %Y") + ': ' + datetime.now().strftime("%I:%M:%S %p"))	
    print(f'device: {device}\t, suffix: {suffix}\n\n')
    
    ## save args to a text file
    args_dict = vars(args)

    with open(filepath + 'args_' + suffix + '.txt','w') as outfile:
        for key, value in args_dict.items(): 
            outfile.write('%s:%s\n' % (key, value))
    return suffix, device, filepath


def create_parser(dir_name):

    parser = argparse.ArgumentParser(description='transformer based approach')
    # data processing related
    parser.add_argument('--hop', type=int, default=2, help='')
    parser.add_argument('--data', type=str, default='WN18RR', dest='dataset', help='choices: FB15K-237, WN18, WN18RR, DDB14_, NELL995')
    # base model related
    parser.add_argument('--d', type=int, default=64, help='feature dimension for each entity and relation')
    parser.add_argument('--m', type=int, default=32, help='maximum neighborhood size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--nh', type=int, default=4, dest='nheads', help='number of attention heads')
    parser.add_argument('--nl', type=int, default=2, dest='nlayers', help='number of attention layers')
    parser.add_argument('--ffn', type=int, default=2, help='multiplier to get dim_feedforward in Transformer block')
    # regularization hyperparams
    parser.add_argument('--wd', type=float, default=0., dest='weight_decay', help='weight decay')
    parser.add_argument('--dr', type=float, default=0.2, dest='dropout', help='dropout rate in attention module')
    # other hyperparams
    parser.add_argument('--log', type=int, default=10, help='log interval')
    parser.add_argument('--ne', type=int, default=100, dest='num_epochs', help='number of epochs')
    parser.add_argument('--b', type=int, default=1024, dest='batch_size', help='number of samples in a minibatch')
    parser.add_argument('--path', type=str, default='c', required=False, help='save all figures and files here')
    parser.add_argument('--cpu', action='store_true', help='when true: use cpu, else, use GPU by default')
    parser.add_argument('--note', type=str, default='c', required=False, help='Notes about trying out a particlar configuration')
    # model variations; choice of functions
    parser.add_argument('--agg', type=str, default = 'concat', help='aggregation function for model; options=[mean, concat]')

    
    args = parser.parse_args()
    if args.d % args.nheads != 0:
        raise ValueError('Feature dimension must be divisible by number of attention heads.')
    if not os.path.isdir('./'+ dir_name + '/'):
        os.mkdir('./'+ dir_name + '/')
    suffix, device, filepath = process_args(args, dir_name)
    print(f'args: {args}')

    return args, suffix, device, filepath
    
