import argparse, os
from datetime import datetime, date, timedelta
from os import listdir
from os.path import isfile, join

def create_parser():
    parser = argparse.ArgumentParser(description='Line graph based approach')

    parser.add_argument('--hop', type=int, default=2, choices=[1,2,3,4], help='number of hops')
    parser.add_argument('--m', type=int, default=32, help='maximum triple for combined triple neighborhood (to control line graph creation time)')
    # model architecture params
    parser.add_argument('--data', type=str, default='WN18RR_v1', dest='dataset')
    parser.add_argument('--ne', type=int, default=10, dest='num_epochs', help='number of epochs')
    parser.add_argument('--d', type=int, default=20, help='feature dimension for each entity and relation')
    parser.add_argument('--nh', type=int, default=4, dest='nheads', help='number of attention heads')
    parser.add_argument('--nl', type=int, default=2, dest='nlayers', help='number of attention layers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., dest='weight_decay', help='weight decay')
    parser.add_argument('--dr', type=float, default=0.2, help='dropout rate in attention module')

    # other hyperparams (not usually changed)
    parser.add_argument('--b', type=int, default=512, dest='batch_size', help='number of samples in a minibatch')
    parser.add_argument('--b_eval', type=int, default=32, help='number of samples in a minibatch')
    parser.add_argument('--log', type=int, default=10, help='log interval')
    parser.add_argument('--nw', type=int, default=6, dest='num_workers', help='number of workers to use in dataloader')
    parser.add_argument('--ffn', type=float, default=2, help='multiplier to get dim_feedforward')
    parser.add_argument('--path', type=str,  help='save all figures and files here', default='c')
    parser.add_argument('--note', type=str,  help='Notes to remember the reason behind trying out a particlar configuration', default='c')
    parser.add_argument('--cpu', action='store_true', help='when true: uses cpu, else uses GPU by default')
    parser.add_argument('--agg', type=str, default = 'concat', help='aggregator function to use in model, options: [mean, concat, mean_lin, svd_agg]')
    parser.add_argument('--nn_train', type=int, default=1, help='number of total samples for each positive sample for training data')
    parser.add_argument('--nn_valid', type=int, default=50, help='number of total samples for each positive sample for validation data')
    parser.add_argument('--nn_test', type=int, default=50, help='number of total samples for each positive sample for test data')
    parser.add_argument('--rp', action='store_true', help='when true: task is relation prediction; otherwise entity prediction')
    parser.add_argument('--bce', action='store_true', help='when true: use BCEloss, otherwise margin based loss [in entity prediction]')
    parser.add_argument('--margin', type=float, default=0.5, help='margin for the margin based ranking loss')
    parser.add_argument('--pe', type=str, default='basic', choices=['basic', 'svd', 'eig'])
    parser.add_argument('--qb', action='store_true', help='when true: use Edge bias for queries; otherwise, only use key and value bias')
    parser.add_argument('--k', type=int, default=8, help='how many first eigenvectors to choose')
    parser.add_argument('--ss', action='store_true', help = 'when true: skip shuffle of data; otherwise shuffle in every epoch')
    parser.add_argument('--s2', action='store_true', help = 'when true: use 2-layer MLP for scoring function')
    parser.add_argument('--reuse', action='store_true', help = 'when true: store and load saved data for neighbors, paths etc')
    parser.add_argument('--hneg', action='store_true', help = 'when true: use hard negative samples')

    ## path related args

    parser.add_argument('--wsum', action='store_true', help = 'when true: use weighted sum of con+path')
    parser.add_argument('--up', action='store_true', help = 'when true: use paths')
    parser.add_argument('--uptr', action='store_true', help = 'when true: use transformer to combine context and paths')
    parser.add_argument('--upmlp', action='store_true', help = 'when true: use mlp to combine context and path')
    parser.add_argument('--p2', action='store_true', help = 'when true: use 2-layer MLP to create paths from rels')
    parser.add_argument('--mpl', type=int, default=4, dest='max_path_len')
    parser.add_argument('--mpc', type=int, default=5, dest='max_path_count')

    parser.add_argument('--gcn', action='store_true', help = 'when true: use gcn')
    parser.add_argument('--gcndr', type=float, default=0.2, help='gcn dropout')
    parser.add_argument('--gcnc', action='store_true', help = 'when true: use gcn and clusters')
    ## AM related args
    parser.add_argument('--nt', action='store_true', help='when True, omit the type encoding H/T/Other and only use entity embedding from us/gcn')
    parser.add_argument('--us', action='store_true', help = 'when true: use signatures for entity')
    parser.add_argument('--usc', action='store_true', help = 'when true: use signatures and match to cluster prototypes')
    parser.add_argument('--k_sig', type=int, default=8, help='how many entity signatures to learn')
    parser.add_argument('--alpha', type=float, default=0.01, help='gradient descent step size')
    parser.add_argument('--beta', type=float, default=10, help='inverse temperature')
    parser.add_argument('--T', type=int, default=4, help='Number of time steps')
    ## neighborhood related
    parser.add_argument('--common', action='store_true', help = 'when true: use common neighbors')
    parser.add_argument('--rsc', action='store_true', help = 'when true: use target rel in scoring function')
    

    args = parser.parse_args()

    if args.d % args.nheads !=0:
        raise ValueError('Embed dim not divisible by number of attention heads.')
    if args.usc and not args.us:
        raise ValueError('Cannot use -usc without -us.')
    
    device = 'cuda'
    if args.cpu:
        device = 'cpu'
    if not os.path.isdir('./results/'):
        os.mkdir('./results/')
    filepath = './results/' + args.path + '/'
    if not os.path.isdir(filepath):
        os.mkdir(filepath)  
    
    
    # find existing args files and increment by 1 to assign new suffix values
    suffix_list = [int(f[5:-4]) for f in listdir(filepath) if isfile(join(filepath, f)) and f.startswith('args_')]
    suffix = max(suffix_list) + 1 if len(suffix_list)>0 else len(suffix_list) + 1
    suffix = str(suffix)

    with open(filepath + 'args_' + suffix + '.txt','w') as outfile:
        for key, value in (vars(args)).items(): 
            outfile.write('%s:%s\n' % (key, value))
    
    print(date.today().strftime('%B %d, %Y') + ': ' + datetime.now().strftime('%I:%M:%S %p'))
    print(f'suffix: {suffix}, Note: {args.note}')
    
    return args, device, suffix, filepath
