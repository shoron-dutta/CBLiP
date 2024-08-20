## Wordnet
# python main.py --data WN18RR_v1 --m 12  --hop 3 --ne 1 --d 20 --lr .005 --nl 3 --b 1024 --ffn 2 --agg mean --path wn_v1
# python main.py --data WN18RR_v2 --agg mean --ne 100 --d 20 --nh 4 --nl 3 --lr .005 --b 1024 --b_eval 32 --path wn_v2
# python main.py --data WN18RR_v3 --m 12 --hop 3 --d 20 --lr .001 --nh 2 --b 3000 --b_eval 32 --agg mean  --path wn_v3 --ne 30
# python main.py --data WN18RR_v4 --m 12 --hop 3 --ne 100 --d 20 --nh 4 --nl 3 --lr .005 --b 1024 --agg mean --path wn_v4 


## Freebase
# python main.py --data fb237_v1 --m 16 --hop 3 --d 20 --nh 2 --nl 2 --lr .0001 --ne 30 --path fb_v1 --b 2048 --b_eval 32 --agg mean 
# python main.py --data fb237_v2 --m 20 --d 40 --common  --lr .001 --b 512 --b_eval 32  --ne 120 --path fb_v2 --agg mean
# python main.py --data fb237_v3 --m 20 --hop 2 --ne 100 --nh 2 --b 2048 --lr .0001 --d 20 --agg concat --path fb_v3
# python main.py --data fb237_v4 --m 20  --ne 30 --d 40 --nh 4 --lr .001 --b 512 --path fb_v4 --agg mean

## Nell
# python main.py --hop 3 --m 24          --data nell_v1 --d 20 --agg mean --lr .005 --b 2048 --path nell_v1 --ne 40
# python main.py --common --hop 2 --m 24 --data nell_v2 --d 20 --agg concat --lr .0001 --b 2048 --ffn 1    --path nell_v2 --ffn 1 --ne 60
# python main.py --common --hop 2 --m 20 --data nell_v3 --d 20 --agg concat --lr .0001 --b 2048 --ffn 1   --path nell_v3 --ffn 1 --ne 100
# python main.py --common --hop 2 --m 20 --data nell_v4 --d 20 --agg concat --lr .0001 --b 2048 --ffn 1  --path nell_v4 --ffn 1 --ne 100
