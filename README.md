## NesT

<img src="./nest.png" width="800px"></img>

This <a href="https://arxiv.org/abs/2105.12723">paper</a> decided to process the image in hierarchical stages, with attention only within tokens of local blocks, which aggregate as it moves up the heirarchy. The aggregation is done in the image plane, and contains a convolution and subsequent maxpool to allow it to pass information across the boundary.

## In collaboration with:
- [Dr. Phil Wang](https://github.com/lucidrains/)

## Usage

```python
import numpy

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 224, 224, 3))

v = NesT(
    image_size = 224,
    patch_size = 4,
    dim = 96,
    heads = 3,
    num_hierarchies = 3,        # number of hierarchies
    block_repeats = (2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
    num_classes = 1000
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")
```
## Todo

- [x] Build model
- [ ] Implement pip installer
- [ ] Implement huggingface streaming dataloaders
- [ ] Implement training scripy

## Author:
- Enrico Shippole

## Citations:
```bibtex
@misc{zhang2021aggregating,
    title   = {Aggregating Nested Transformers},
    author  = {Zizhao Zhang and Han Zhang and Long Zhao and Ting Chen and Tomas Pfister},
    year    = {2021},
    eprint  = {2105.12723},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
```bibtex
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.5.0},
  year = {2020},
}
```
```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```
