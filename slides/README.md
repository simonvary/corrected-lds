# Talk figures

The result figures of the talk on the paper, numbered in the order the talk shows them. They
are built the way the paper's are (`paper/_harness.py`): `compute()` writes
`plotdata/<name>.json`, and `plot()` draws `figures/<name>.pdf` and a 300-dpi `.png`.

```bash
python slides/fig_slides1.py        # render from committed plotdata: seconds, no banks
python slides/fig_slides1.py --recompute --banks-root banks    # about a minute on CPU
```

| script | draws | numbers from |
|---|---|---|
| `fig_slides1.py` | Pearson LDS against retrains `N = M·R` for naive R = 5, 10, 20, 40, each with the ceiling it stalls at, labelled with its fraction of the noise-free LDS; `_step1` to `_step4` add the lines one at a time without the fractions, and `_corrected` adds the corrected R = 2 line | cifar10 bank: `pool`, one model of `pool_tda_ekfac` |
| `fig_slides2.py` | RMSE to the five-seed LDS ρ₅: naive R = 5 against reconstructed from R = 2, TRAK on CIFAR-10, QNLI and WikiText-2 | paper Figure 2(b), `paper/plotdata/fig2_panels.json` |
| `fig_slides3.py` | corrected LDS against attributor ensemble size E, measured and forecast from K members | paper Figure 2(c), the same file |

The last two re-read the paper's committed plotdata on `--recompute`, so they always show
the paper's numbers. The talk includes the PDFs. After a change, copy `figures/*.pdf` into the
talk's `figures/` folder.
