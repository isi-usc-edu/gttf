python3 train_torch_gcn.py --epochs=300 --gcn_dataset='ind.cora' --model='simple_gcn.SGC' --lr=1e-2 --l2_reg=1e-3\
  --model_kwargs='{"k":2}'