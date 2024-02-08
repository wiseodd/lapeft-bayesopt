# `lapeft-bayesopt`: Discrete Bayesian Optimization with LLM + PEFT + Laplace Approximation

This is the accompanying library of the paper [_A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules?_](https://arxiv.org/abs/2402.05015).

If you use this library, please cite using the following bib entry:

```
@article{kristiadi2024sober,
  title={A Sober Look at {LLMs} for Material Discovery: {A}re They Actually Good for {B}ayesian Optimization Over Molecules?},
  author={Kristiadi, Agustinus and Strieth-Kalthoff, Felix and Skreta, Marta and Poupart, Pascal and Aspuru-Guzik, Al\'{a}n and Pleiss, Geoff},
  journal={arXiv preprint arXiv:2402.05015},
  year={2024}
}
```

## Table of Contents
 1. [Setup](#setup)
 2. [Warmup: Using LLMs as _fixed_ feature extractors](#fixed-feature)
 3. [Using finetuned LLMs as surrogates](#finetuning)


<a id="setup"></a>
## Setup

Best done in a fresh conda/mamba environment (Python < 3.12). Note that the ordering below is important.

1. Install PyTorch (with CUDA; version 2+ is supported): <https://pytorch.org/get-started/locally/>
2. Install a specific branch of laplace-torch: `pip install git+https://github.com/aleximmer/Laplace.git@mc-subset2`
3. Install a specific version of ASDL (to compute Hessians): `pip install git+https://github.com/wiseodd/asdl.git@dev`
4. Clone and install this repo:
```
git clone git@github.com:wiseodd/lapeft-bayesopt.git
cd lapeft-bayesopt
pip install -e .
```

<a id="fixed-feature"></a>
## Warmup: Using LLMs as _fixed_ feature extractors

**Full example:** `examples/run_fixed_features.py`

The simplest way to incorporate LLMs into BO surrogates is by viewing them as fixed feature extractors.
Given a data point $x \in \mathcal{D}\_\mathrm{cand}$ from the pool of candidates $\mathcal{D}\_\mathrm{cand} = \\{x_1, \dots, x_n\\}$ we want to find the best from, we wrap it in a textual prompt $c(x)$ and then do a forward pass over the LLM and take the last transformer embedding which has shape `(seq_len, embd_dim)`.
Then, we aggregate it by e.g. averaging over the sequence dimension to get a feature vector for $h(x)$ with shape `(embd_dim,)`.
Doing this for all $x$'s, we can then do the usual discrete BO loop with standard surrogate functions like GPs or Bayesian NNs over the candidates $\mathcal{D}_\mathrm{cand}$.

This package provides an easy way to do the transformation from $x$ to $h(x)$.
Here are the steps:

1. We assume that your dataset is a pandas dataframe. Inherit `lapeft_bayesopt.problems.DataProcessor`. Example (from `examples/data_processor.py`):
```python
import lapeft_bayesopt.problems.DataProcessor

class RedoxDataProcessor(DataProcessor):
    """
    Pandas dataframe spec:
    ----------------------
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 6 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   Ered                   1407 non-null   float64
    4   HOMO                   1407 non-null   float64
    5   Gsol                   1407 non-null   float64
    6   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered over a list of molecules in SMILES.
    """
    def __init__(self, prompt_builder, tokenizer):
        # num_outputs = 1 since this is a single-objective problem
        super().__init__(prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer)

        # We must specify the four properties below
        # -----------------------------------------
        # `x_col` is the column name of your x
        self.x_col = 'SMILES'
        # `target_col` is the pandas column name of the property that we want to optimize
        self.target_col = 'Ered'
        # `obj_str` is the textual description of that property (useful for prompting the LLM later)
        self.obj_str = 'redox potential'
        # `maximization` is whether we want to maximize or minimize the property
        self.maximization = False

    def _get_columns_to_remove(self) -> List[str]:
        # List all columns of your dataset! We will remove all of them after we preprocess the dataset using Huggingface (we only need the resulting `input_ids` and `labels`)
        return ['Entry Number', 'File Name', 'SMILES', 'HOMO', 'Ered', 'Gsol', 'Absorption Wavelength']
```

2. Next, we create prompting schemes $c(x)$. Here's an example (from `examples/prompting.py`):

```python
from lapeft_bayesopt.problems.prompting import PromptBuilder

class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str):
        self.kind = kind

    def get_prompt(self, smiles_str: str, obj_str: str) -> str:
        if self.kind == 'completion':
            return f'The estimated {obj_str} of the molecule {smiles_str} is: '
        elif self.kind == 'just-smiles':
            return smiles_str
        else:
            return NotImplementedError

```

3. Then, we need the LLM feature extractor itself. This package has some ready-made ones, e.g., `lapeft_bayesopt.foundation_models.t5.T5Regressor`. Feel free to follow that example to create your own.

4. Then, we can start extracting the LLM features from $\mathcal{D}_\mathrm{cand}$. See the `load_features` method in `examples/run_fixed_features.py`.

5. Finally, we can do the discrete BO loop using those features (cache provided in `examples/data/cache`). See `examples/run_fixed_features.py` for a complete, self-contained example. Note that, at this point, we can use any BO algorithm and surrogate function. E.g. we can just use BoTorch for the BO loop.


<a id="finetuning"></a>
## Using finetuned LLMs as surrogates

**Note:** Please check the previous section first since we will reuse some objects here.

**Full example:** `examples/run_finetuning.py`

We can go one step further by making $h(x)$ trainable.
This can be done by attaching a PEFT method (LoRA, PrefixTuning, Adapter, etc) to the frozen LLM.
Then, we train and do a Laplace approximation on the PEFT's and regression head's weights.

To do this, we can use the surrogates provided in `lapeft_bayesopt.surrogates`.
Currently LoRA is supported and it is very easy to support other PEFT methods, using Huggingface's `peft` library.

See `examples/run_finetuning.py` for an example. It's actually quite simple to use!

1. First, define the base PEFT-infused LLM in a function so that it is freshly initialized at each call. (Useful since at each BO iteration, the surrogate model is retrained.)
```python
def get_model():
    # Load a foundation model with a regression head attached
    model = T5Regressor(
        kind='GT4SD/multitask-text-and-chemistry-t5-base-augm',
        tokenizer=tokenizer
    )

    # Attach LoRA or any other PEFT on the foundation model
    target_modules = ['q', 'v']
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias='none',
        # This is necessary so that the regression head is also trained
        modules_to_save=['head'],
    )
    lora_model = get_peft_model(model, config)

    # For some reason HF's peft duplicates the head. So we need to "detach" the one that is unused
    for p in lora_model.base_model.head.original_module.parameters():
        p.requires_grad = False

    return lora_model
```

2. Then, we can configure the training and the Laplace approximation of this PEFT surrogate. For full options, check out `lapeft_bayesopt.utils.configs.LaplaceConfig`
```python

# Config for the Laplace approx over PEFT
config = LaplaceConfig(
    noise_var=0.001,
    hess_factorization='kron',
    subset_of_weights='all',
    marglik_mode='posthoc',
    prior_prec_structure='layerwise'
)
```

3. We then initialize a training dataset, e.g. via random sampling from the candidate set (a pandas dataframe). The training dataset should be a list of pandas rows or dicts. Don't forget to remove that row from the original dataset $\mathcal{D}_\mathrm{cand}$. We can use `lapeft_bayesopt.utils.helpers.pop_df()` to do so.
```python
dataset_train = []
while len(dataset_train) < n_init_data:
    idx = np.random.randint(len(pd_dataset))
    # Make sure that the optimum is not included
    if pd_dataset.loc[idx][OBJ_COL] >= ground_truth_max:
        continue
    dataset_train.append(helpers.pop_df(pd_dataset, idx))
```

4. Next, create the surrogate.
```python

# Create the surrogate model based on the LLM+PEFT regressor above
model = LAPEFTBayesOptLoRA(
    get_model, dataset_train, data_processor, laplace_config=config
)
```

5. At each BO iteration, we preprocess the candidate $x$'s and infer the posterior mean and variance of the Laplace approximation over PEFT (LAPEFT!) to compute the acquisition function.
```python
# Preprocess D_cand (`dataset`) so that we can make predictions over it
dataloader = data_processor.get_dataloader(pd_dataset, batch_size=16, shuffle=False)

# Make prediction over D_cand, get means and vars, compute the acqf
acq_vals = []
for data in dataloader:
    posterior = model.posterior(data)
    f_mean, f_var = posterior.mean, posterior.variance
    acq_vals.append(thompson_sampling(f_mean, f_var))
acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
```

6. We pick the $x$ that maximizes the acquisition function and remove it from the candidate set. That $x$ is represented by a pandas row that we popped from the candidate set (a pandas dataframe).
```python
# Pick an x (a row in the current pandas dataset) that maximizes the acquisition
idx_best = torch.argmax(acq_vals).item()
new_data = helpers.pop_df(pd_dataset, idx_best)

# Update the current best y
if new_data[OBJ_COL] > best_y:
    best_y = new_data[OBJ_COL]
```

7. Finally, we just feed this new data point to the surrogate object. It will append it to the training set and retrain the surrogate for the next iteration.
```python
# Update surrogate using the new data point
model = model.condition_on_observations(new_data)
```
