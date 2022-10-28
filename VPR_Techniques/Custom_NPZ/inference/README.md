# Required Files

This folder should contain 2 npz files:
- descs.npz
- times.npz

### NPZ Structure

```
{
    "Path0": Embedding0,
    "Path0": Embedding0,
}
```

where `Path`s should be in the format `<dataset_name>/<query_or_ref_dir>/<image_index>.jpg`. Note that the DatasetName should not include the full path, use relative path for portability of npz files e.g. `CrossSeason_CoHOG_Dataset/query/000000.jpg`
