import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt

# Foundation models etc from lapeft_bayesopt
from lapeft_bayesopt.foundation_models.t5 import T5Regressor
from lapeft_bayesopt.foundation_models.utils import get_t5_tokenizer
from lapeft_bayesopt.utils.acqf import thompson_sampling
from lapeft_bayesopt.utils.configs import LaplaceConfig
from lapeft_bayesopt.utils import helpers

# The PEFT surrogate
from lapeft_bayesopt.surrogates.lora import LAPEFTBayesOptLoRA

# We also need Huggingface's peft for the finetuning
from peft import LoraConfig, get_peft_model

# Our self-defined problems, using the format provided by lapeft-bayesopt
from data_processor import RedoxDataProcessor
from prompting import MyPromptBuilder


def main():
    dataset = {
        'pd_dataset': pd.read_csv('data/redox_mer.csv'),
        'smiles_col': 'SMILES',
        'obj_col': 'Ered',
        'maximization': False,
    }

    prompt_builder = MyPromptBuilder(kind='just-smiles')
    tokenizer = get_t5_tokenizer('GT4SD/multitask-text-and-chemistry-t5-base-augm')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_processor = RedoxDataProcessor(prompt_builder, tokenizer)

    results = run_bayesopt(dataset, tokenizer, data_processor, n_init_data=10, T=30, randseed=9999)

    # Plot
    t = np.arange(len(results))
    plt.axhline(dataset['opt_val'], color='black', linestyle='dashed')
    plt.plot(t, results)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Objective ($\downarrow$)')
    plt.show()


def run_bayesopt(dataset, tokenizer, data_processor, n_init_data=10, T=30, randseed=1):
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    pd_dataset = dataset['pd_dataset']
    SMILES_COL = dataset['smiles_col']
    OBJ_COL = dataset['obj_col']
    MAXIMIZATION = dataset['maximization']

    # Turn into a maximization problem if necessary
    if not MAXIMIZATION:
        pd_dataset[OBJ_COL] = -pd_dataset[OBJ_COL]
    ground_truth_max = pd_dataset[OBJ_COL].max()

    dataset_train = []
    while len(dataset_train) < n_init_data:
        idx = np.random.randint(len(pd_dataset))
        # Make sure that the optimum is not included
        if pd_dataset.loc[idx][OBJ_COL] >= ground_truth_max:
            continue
        dataset_train.append(helpers.pop_df(pd_dataset, idx))

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

        # For some reason HF's peft duplicates the head. So we need "detach" the one that is unused
        for p in lora_model.base_model.head.original_module.parameters():
            p.requires_grad = False

        return lora_model

    # Config for the Laplace approx over PEFT
    config = LaplaceConfig(
        noise_var=0.001,
        hess_factorization='kron',
        subset_of_weights='all',
        marglik_mode='posthoc',
        prior_prec_structure='layerwise'
    )

    # Create the surrogate model based on the LLM+PEFT regressor above
    model = LAPEFTBayesOptLoRA(
        get_model, dataset_train, data_processor, dtype='float32', laplace_config=config
    )

    # Prepare the BO loop
    best_y = pd.DataFrame(dataset_train)[OBJ_COL].max()
    pbar = tqdm.trange(T, position=0, colour='green', leave=True)
    pbar.set_description(
        f'[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}]'
    )

    # To store the logged best f(x) over time
    trace_best_y = [helpers.y_transform(best_y, MAXIMIZATION)]

    # BO iteration
    for t in pbar:
        # Preprocess D_cand (`dataset`) so that we can make predictions over it
        dataloader = data_processor.get_dataloader(pd_dataset, batch_size=16, shuffle=False)

        # Make prediction over D_cand, get means and vars, compute the acqf
        acq_vals = []
        sub_pbar = tqdm.tqdm(
            dataloader, position=1, colour='blue',
            desc='[Prediction over dataset]', leave=False
        )
        for data in sub_pbar:
            posterior = model.posterior(data)
            f_mean, f_var = posterior.mean, posterior.variance
            acq_vals.append(thompson_sampling(f_mean, f_var))
        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()

        # Pick a molecule (a row in the current pandas dataset) that maximizes the acquisition
        idx_best = torch.argmax(acq_vals).item()
        new_data = helpers.pop_df(pd_dataset, idx_best)

        # Update the current best y
        if new_data[OBJ_COL] > best_y:
            best_y = new_data[OBJ_COL]

        # Remember that the cached features are always in maximization format.
        # So here, we transform it back if necessary.
        pbar.set_description(
            f'[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, '
            + f'curr f(x) = {helpers.y_transform(new_data[OBJ_COL], MAXIMIZATION):.3f}]'
        )

        # Update surrogate using the new data point
        model = model.condition_on_observations(new_data)

        # Log the best f(x)
        trace_best_y.append(helpers.y_transform(best_y, MAXIMIZATION))

    return trace_best_y


if __name__ == '__main__':
    main()
