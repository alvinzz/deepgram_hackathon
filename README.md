# deepgram_hackathon

Work done for the DeepGram deep learning hackathon. 

Attempts to find a latent representation for reconstructing EEG data using convnets, then use this latent represenation to classify and reconstruct images.

Only EEG-to-EEGnet has been trained.

Rest of training to be done on a p2.xlarge AWS ECS instance.

Made AMI (Amazon Machine Image) for TensorFlow 1.0, Python 3.6, CUDA 8.0, CUDNN 5.1, available on US-West-1 (NorCal) servers as ami-822973e2.

=================

Data has not been added to this repo.
Training and validation folders should be added to EEGdata.
Cleaned data available at:
* https://kur.deepgram.com/data/mind/stanford-mind-reading-train.tar.gz
* https://kur.deepgram.com/data/mind/stanford-mind-reading-validate.tar.gz

Original dataset:
* https://exhibits.stanford.edu/data/catalog/tc919dd5388
