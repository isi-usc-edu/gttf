python3 train_torch_sage.py --data_dir=../../../datasets --dataset=amazon --epochs=10 --batch_size=512 --fanouts=10x10 --lr=0.0007 --hidden_dims=256,256 --dropout=0.35 --num_workers=6 --device=cuda