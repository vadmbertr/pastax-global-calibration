Material of the [JAXATHON 2025](https://github.com/Diff4Earth/ige-jaxathon-2025) organized at IGE.

The following repository is made of five notebooks separated into two directories:
- data_loading:
	- [experimental.ipynb](data_loading/experimental.ipynb): some experiments, with notes on what worked and what did not,
	- [timings.ipynb](data_loading/timings.ipynb): produce data loading timings.
- calibration:
	- [timings.ipynb](calibration/timings.ipynb): produce calibration timings,
	- [results.ipynb](calibration/results.ipynb): runs calibation and plots the loss and the Smagorinsky constant across iterations,
	- [timings_multi_gpus.ipynb](calibration/timings_multi_gpus.ipynb): same as timings.ipynb, but using several GPUs,
	- [results_multi_gpus.ipynb](calibration/results_multi_gpus.ipynb): same as results.ipynb, but using several GPUs.

See [pastax_calibration.md](https://github.com/Diff4Earth/ige-jaxathon-2025/blob/main/projects/pastax_calibration/pastax_calibration.md) for more details on the objectives and outcomes of the experiments.
