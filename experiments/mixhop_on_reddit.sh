# 0.944718
python3 train_tf_reddit.py --model='mixhop.MixHopWithFCClassifier' --epochs=10 --fanout=5x5x5x5 --model_kwargs='{"layer_dims": [[80,80,80], [80,80,80]]}' --eval_every=10