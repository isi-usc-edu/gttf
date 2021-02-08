# 0.949109
python3 train_tf_reddit.py --model='kipf_gcn.KipfGCN' --epochs=10 --fanout=25x10 \
    --model_kwargs='{"hidden_dim": 256}' --eval_every=10