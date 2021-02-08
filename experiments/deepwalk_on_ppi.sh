ls ~/data/asymproj || echo `mkdir -p ~/data/asymproj; cd ~/data/asymproj; curl http://sami.haija.org/graph/datasets.tgz > datasets.tgz; tar zxvf datasets.tgz`
python3 train_torch_skipgram.py --dataset_dir=~/data/asymproj/datasets/ppi --steps=10000 --lr 0.01 --d 128 --neg_samples=20 --fanout=1x1x1x1x1x1x1 --batch_size=1000
# 0.905141