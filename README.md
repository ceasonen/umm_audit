# Auditing Transfer from Visual QA to Image Generation

UMM Audit examines how a removable update trained on visual question answering affects image generation in six unified multimodal models. It measures whether answering gains depend on the input image, whether activating the update changes generation, and whether swapping in the update from a different prompt transfers its attributes.

[Project site](https://ceasonen.github.io/umm_audit/) · [Scoring demo](https://ceasonen.github.io/umm_audit/audit/) · [Results CSV](site/data/results.csv)

## Browse the results

The website shows paired counting and spatial results for BLIP3-o, JanusFlow, UniWorld-V1, DiMOO, Janus-Pro and Show-o2. The interactive table switches between trained-versus-frozen scores and norm-matched conditioning swaps. The demo uses two BLIP3-o outputs and keeps draft ratings in the browser, with JSON draft import/export and CSV export for completed ratings.

To serve the project site locally, run:

```sh
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000/`. Run the scoring checks with `node --test tests/*.test.cjs`.

## Read the experiment code

| Task | Code |
| --- | --- |
| Attach the removable rank-limited update | [Adapter](research/src/skillbridge/sidecar.py), [training loop](research/src/skillbridge/training.py) |
| Test whether answers depend on the image | [Source controls](research/scripts/run_original_core_source_audit.py), [answer measurement](research/src/skillbridge/source_measurement.py) |
| Compare trained and frozen generation | [Endpoint runner](research/scripts/run_mechanism_endpoints.py), [paired evaluation](research/src/skillbridge/cross_backbone_evaluation.py) |
| Swap a donor update and measure both attributes | [Conditioning swap runner](research/scripts/run_condition_interchange.py), [interventions](research/src/skillbridge/mechanism_sequence.py), [aggregation](research/src/skillbridge/donor_replication.py) |
| Compare with standard additive LoRA | [Baseline](research/src/skillbridge/baseline.py) |

The release includes the analysis code and configuration references used to inspect the pipeline. Running the full experiment suite requires the upstream model repos and weights, evaluators, datasets, experiment artifacts and GPU environment referenced in the scripts.

## License

Project code is available under [MIT](LICENSE). The website adapts [Stellar by HTML5 UP](https://html5up.net/stellar), licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Bundled third-party code retains its upstream notices, including the [Show-o2 transport code](research/src/skillbridge/vendor/showo2_transport/README.md).
