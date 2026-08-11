`sample_squad_tiny.json` is a small synthetic file (15 passages) in SQuAD format,
used only to sanity-check the pipeline (`src/data/loader.py`) without needing the
full dataset. It is not used for any reported experiment results.

For actual experiments, download SQuAD (train or dev split) from
https://rajpurkar.github.io/SQuAD-explorer/ and place the json file here, then
point `config/default.yaml`'s `data.squad_path` at it.
