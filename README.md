# Implicit neural representations for 4D flow MRI
This page contains a Pytorch implementation of the
[paper](https://arxiv.org/pdf/2302.12835.pdf) "Implicit Neural Representations for 
unsupervised super-resolution and denoising of 4D flow MRI".

## Under development

## Data description
Each case to process is expected to be a .h5 (or .hdf5) file with the following structure: 
```bash
<h5 file>
├── meta
│   ├── dt
│   └── spacing
├── obs
│   ├── t
│   ├── xyz
│   ├── u
│   ├── v
│   └── w
└── wall
    └── xyz
```
`meta`: metadata keys
- `dt`: time spacing between 4D flow frames
- `spacing`: 4D flow voxel spacing (list of 3 numbers)

`obs`: observation keys
- `t`: time coordinates array (shape `[Nt x 1]`)
- `xyz`: space coordinates array (shape `[Nf x 3]`)
- `u`: x-component of velocity vectors (array of `[Nf x Nt]`)
- `v`: y-component of velocity vectors (array of `[Nf x Nt]`)
- `w`: z-component of velocity vectors (array of `[Nf x Nt]`)

`wall`: wall keys
- `xyz`: space coordinate array (shape `[Nw x 3]`)\

>**Note**: the order of coordinates and velocity arrays matters! Sortings should correspond.


## Cite this work

If you use this code for academic research, please cite the following paper:

```
@article{saitta2023implicit,
  title={Implicit neural representations for unsupervised super-resolution and denoising of 4D flow MRI},
  author={Saitta, Simone and Carioni, Marcello and Mukherjee, Subhadip and Sch{\"o}nlieb, Carola-Bibiane and Redaelli, Alberto},
  journal={arXiv preprint arXiv:2302.12835},
  year={2023}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section. 
Please note that this repository is still under development and some features may be missing.
