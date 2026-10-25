# Step 5: Missing-phase depth and statistics

`summarize.py` reads `judged/rows.csv` and prints, for the last generation (or `--gen`):

- per run, adaptive vs passive paired by seed: the fraction the reward claims stable, the fraction the held-out judges verify, and the Wilcoxon signed-rank p;
- the depth of missing phases (reward hull minus complete hull, eV/atom): the median and the share of candidates deeper than `--deep`;
- the verified fraction at every generation;
- with `--compare`, two runs (for example a defense and its baseline) across seeds, with the Mann-Whitney p.

Decide the metric, the seeds and the test before you look (see "Checking your results" in the top-level README).

## Command

```bash
python summarize.py [--rows FILE] [--gen N] [--compare RUN_A RUN_B] [--lenient] [--deep 0.1] [--min-rows N]
```

## Outputs

Printed tables, plus `judged/seed_level.csv`.

## Options

| Option | Meaning |
|---|---|
| `--rows FILE` | input rows (default `$MATSCI_SCRATCH/judged/rows.csv`) |
| `--gen N` | generation to compare (default: the last one) |
| `--compare RUN_A RUN_B` | compare two runs across seeds |
| `--lenient` | count +U-regime candidates as verified when the judge says stable |
| `--deep X` | depth threshold in eV/atom (default 0.1) |
| `--min-rows N` | drop seed cells with fewer judged rows |
