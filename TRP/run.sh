
# python main.py --data FB15K-237 --hop 1 --d 20 --agg concat --m 20 --lr 0.01    --nh 4 --nl 3 --ffn 2 --wd 1e-7 --dr 0.2 --ne 40 --b 512  --path fb_237_exp     --note 'Experiment in FB15K-237'
# python main.py --data FB15k     --hop 1 --d 20 --agg concat --m 20 --lr 0.01    --nh 4 --nl 2 --ffn 2 --wd 0.0  --dr 0.2 --ne 20 --b 256  --path fb_1345_exp    --note 'Experiment in FB15k'
# python main.py --data DDB14_    --hop 3 --d 80 --agg mean   --m 24 --lr 0.01    --nh 4 --nl 2 --ffn 2 --wd 5e-4 --dr 0.2 --ne 40 --b 3000 --path ddb_exp        --note 'Experiment in DDB14'
# python main.py --data NELL995   --hop 3 --d 20 --agg concat --m 32 --lr 0.008   --nh 4 --nl 2 --ffn 2 --wd 5e-4 --dr 0.2 --ne 20 --b 2048 --path nell_exp       --note 'Experiment in NELL995'
# python main.py --data WN18RR    --hop 3 --d 64 --agg mean   --m 32 --lr 0.01    --nh 2 --nl 3 --ffn 2 --wd 1e-4 --dr 0.2 --ne 50 --b 256  --path wn_11_exp      --note 'Experiment in WN18RR'
# python main.py --data wn18      --hop 1 --d 32 --agg concat --m 16 --lr 0.001   --nh 4 --nl 2 --ffn 2 --wd 5e-3 --dr 0.2 --ne 20 --b 4096 --path wn_18_exp      --note 'Experiment in WN18'
