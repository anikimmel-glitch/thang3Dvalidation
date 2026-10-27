# Third-party source

- `cadrille/cadrille.py`: unchanged from col14m/cadrille tree
  `d72acc687273d31d62afe62eb9ded8b66b835321`.
  `render.py` extracts `mesh_to_image` from the same tree's `dataset.py` and
  supplies its required imports, without PyTorch3D training dependencies.
  Repository: https://github.com/col14m/cadrille
- `cad_recode/model.py`: extracts `FourierPointEncoder` and `CADRecode` from
  filaPro/cad-recode `demo.ipynb`, tree
  `03e3262119b38939feaa44b8368ad8db99243d47`, with inference imports.
  Repository: https://github.com/filaPro/cad-recode
- `cadreasoner/visualization.py`: unchanged from CADReasoner tree
  `a282d0998b9400ae0ef1082c589a962471f2ec58`.
  Repository: https://github.com/zhemdi/CADReasoner

The supplied identifiers are Git tree IDs observed during source retrieval,
not asserted to be commit IDs. `source_hashes.json` records the exact included
Python files. Original source licenses are retained in each directory.
Checkpoint revisions and model-card licensing metadata are separate in
`models.lock.json`. No third-party weights are included here.
