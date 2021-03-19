### Re-producing results

```
# navigate into this folder, assuming you start in the [PROJECT] folder.
# [PROJECT]/figures
cd figures
pip install palettable, seaborn
python run.py

# outputs all the images into `[PROJECT]/figures/figures/`
# outputs intermediate tsvs into `[PROJECT]/figures/files/`
# outputs all tables into stdout (they're just printed.)
```

### Notes

- Assumes that the data is in /results/..." (data available from my google drive, ~ 2GB)
```
# download `results.zip` from this drive: (or run the code yourself, but that'll take a bit).
https://drive.google.com/drive/folders/1JaTSZqU0IfC4bwWkb3EYF3QRgvMhyCp1?usp=sharing

# unzip
unzip results.zip
mv results [PROJECT]/
```

- In these files, 'weak' means 'spurious' and 'strong' means 'target'.
- We used jupyter notebooks to develop these plots, and refactored this into a pipeline so its easy to run.