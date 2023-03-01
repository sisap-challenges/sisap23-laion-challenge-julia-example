# SISAP23 LAION2B Challenge: similarity search example on Julia

This repository is a working example for the challenge <https://sisap-challenges.github.io/>, working with Julia and GitHub Actions, as specified in Task's descriptions.


## Steps for running
It requires a working installation of Julia (verified with v1.8), which can be downloaded from <https://julialang.org/downloads/>, and an installation of the git tools. You will need internet access for cloning and downloading datasets.

Clone the repository

```bash
git clone https://github.com/sisap-challenges/sisap23-laion-challenge-julia-example/
```

instantiate the directory
```bash
cd sisap23-laion-challenge-julia-example
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Wait for the environment initialization (i.e., installing packages). Run the example

```bash
julia --project=. -t auto src/run.jl
```

This will run the LAION2B 100K subset of 768-dimensional vector embeddings using 10k public queries.

The running will generate a `result` directory that will be used to create scores. 

## Evaluation

You can use an example evaluation (commented). Still, the official one is made with <https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation>, already integrated into the repository as a submodule (also working with GitHub Actions).

To run the official evaluation, you should install its dependencies (python3, matplotlib, and h5py) and then run on the repository's root directory.

```bash
python eval/eval.py
```

Please visit <https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation> for more details.

## How to take this to create my own
You can fork this repository and polish it to create your solution or use it to see how input and output are made to adapt it to your similarity search pipeline. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

Please check the workflow that controls the continuous integration <https://github.com/sisap-challenges/sisap23-laion-challenge-julia-example/blob/main/.github/workflows/ci.yml>. This is a core part of your solution in the SISAP23 challenge (it will ensure that your solution actually works for a given infrastructure setup).

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap23-laion-challenge-julia-example/actions>


## Python + Faiss
You can find the Faiss example on <https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example>

 
