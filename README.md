# GraphNVP: An Invertible Flow Model for Generating Molecular Graphs

<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/graph-nvp/blob/master/assets/fig_interpolation.png" width="800" /> 
</p>

The paper is available on arXiv, [https://arxiv.org/abs/1905.11600](https://arxiv.org/abs/1905.11600).

## Citation
If you find our work useful in your research, please consider citing:

```
@article{kaushalya2019graphnvp,
  title={GraphNVP: An Invertible Flow Model for Generating Molecular Graphs},
  author={Kaushalya, Madhawa and Katushiko,  Ishiguro and Kosuke, Nakago and Motoki, Abe},
  journal={arXiv preprint arXiv:1905.11600},
  year={2019}
}
```


## Dependencies
1. Python 3.6+
1. Chainer<=5.2.0 (Note: code may not work with chainer>=6.0.0)
1. cupy<=5.2.0 (Note: please install the same version with chainer)
1. chainer-chemistry==0.5.0
1. rdkit (release 2017.09.3.0) [Check [chainer-chemistry](https://github.com/pfnet-research/chainer-chemistry) for more information]
1. CUDA-Aware MPI (Only for running on multiple GPUS using ChainerMN. Check [ChainerMN installation guide](https://docs.chainer.org/en/stable/chainermn/installation/guide.html) for more information.)

Example instllation

```
conda install -c rdkit rdkit==2017.09.3.0
pip install -r requirements.txt
# please modify XX into your system's CUDA version
# pip install cupy-cudaXX==5.2.0
pip install cupy-cuda100==5.2.0
# When you want to use ChainerMN (Multi-GPU training)
pip install mpi4py
```

Tested datasets
* QM 9
* Zinc 250k

## Pre-trained models

Pre-trained model files are uploaded. Please download and place them to `models` directory.

 - https://drive.google.com/drive/folders/1bYpPT8jcy3PePBh8_Pp9vGUraBwOVN38

## How to run code

### Dataset preparation

```bash
cd data
# Download and preprocess QM9 dataset
python download_data.py --data_name=qm9
# Download and preprocess ZINC-250k dataset
python download_data.py --data_name=zinc250k
```
We use the same train / validation split used by Kusner et al. ([Grammar VAE](https://github.com/mkusner/grammarVAE))

### Training

- QM9
```bash
python train_model.py -f qm9_relgcn_kekulized_ggnp.npz -b 256 -x 200 --gpu 0 --num_node_masks 9 --num_channel_masks 9 \
  --num_node_coupling 36 --num_channel_coupling 27 --num_atom_types 4 --apply_batch_norm True --node_mask_size 15 \
  --debug True --additive_transformations --save_dir=results/qm9 --learn_dist yes
```

- Zinc250k
```bash
python train_model.py -f zinc250k_relgcn_kekulized_ggnp.npz --data_name=zinc250k --num_atoms=38 -b 128 -x 200 --gpu 0 \
  --num_node_masks 38 --num_channel_masks 38 --num_node_coupling 38 --num_channel_coupling 38 --num_atom_types 9 \
  --apply_batch_norm True --node_mask_size 15 --debug True --additive_transformations \
  --save_dir=results/zinc250k --learn_dist yes
```

For _multi-GPU training_ please check `scripts/train_qm9_chainermn.sh` and `scripts/train_zinc250k_chainermn.sh`.

### Evaluation (Generating molecules with a trained model)

A pre-trained model along with hyperparameters is available.
Please refer "Pre-trained models" section.

- QM9

Executing the bash script `generate.sh` will generate molecules.

```bash
python generate.py -snapshot graph-nvp-final.npz \
--gpu -1 \
--data_name qm9 \
--data_dir data \
--hyperparams-path graphnvp-params.json \
--batch-size 1000 \
--model_dir models/qm9 \
--temperature 0.8 \
--delta 0.05 \
--n_experiments 1
```


- Zinc250k

```bash
python generate.py -snapshot graph-nvp-final-new.npz \
--gpu -1 \
--data_name zinc250k \
--data_dir data \
--hyperparams-path graph-nvp-new-params.json \
--batch-size 1000 \
--model_dir models/zinc-250k \
--temperature 0.75 \
--delta 0.05 \
--n_experiments 1 \
--molecule_file zinc250k_relgcn_kekulized_ggnp.npz
```

## Property optimization

<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/graph-nvp/blob/master/assets/fig_optimization.png" width="600" /> 
</p>

 - QM9 example

```bash
python optimize_property.py -snapshot graph-nvp-final.npz \
 --hyperparams_path graphnvp-params.json \
 --batch_size 1000 \
 --model_dir models/qm9 \
 --data_dir data \
 --molecule_file qm9_relgcn_kekulized_ggnp.npz \
 --temperature 1.0 \
 --delta 0.5 \
 --img_format png \
 --property_name qed \
 --property_model qed_model.pkl 
```
