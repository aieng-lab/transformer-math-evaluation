from datasets import DatasetDict, Dataset


def split(path, size=300000):
    dsd = DatasetDict.load_from_disk(path)
    train = dsd['train'].to_pandas()
    l = len(train)
    for ii, i in enumerate(range(0, l, size)):
        # Use iloc to access the rows within the current slice
        slice_df = train.iloc[i:i + size]
        print(slice_df)
        ds = Dataset.from_pandas(slice_df, preserve_index=False)
        ds.save_to_disk(path + '-splits/%d' % ii)
    dsd['val'].select(list(range(size))).save_to_disk(path + '-splits/test')


split('../data/arqmath-training')