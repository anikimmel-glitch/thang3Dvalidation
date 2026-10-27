# Private run manifest

Supply JSON with `files` and `bin_labels`. Each file row needs:

```json
{
  "pilot_id": "p0000",
  "file": "source-solid.step",
  "sha256": "64-character source SHA-256",
  "bytes": 123456,
  "source_group": "stable connected source/hash group",
  "role": "development",
  "complexity_bin": 2,
  "faces": 42,
  "seed": 20261027
}
```

All file names are basenames relative to the input directory. Pilot IDs, source
groups, and exact source hashes must be distinct. The current private manifest
also retains creator links and license-review status; these are not model inputs.
`pilot.json` is ignored by Git and supplied separately through authorized Sagey
storage. Do not replace a frozen manifest in place after evaluating results.
