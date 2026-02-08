import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TilesSimCLRDataset(Dataset):
    # датасет для simclr - берем один тайл и делаем 2 аугментации
    def __init__(self, csv_path, split, transform):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["tile_path"]).convert("RGB")

        # две разные вьюхи одной картинки
        v1 = self.transform(img)
        v2 = self.transform(img)

        return v1, v2


class PairsDataset(Dataset):
    # пары query/tile + мета
    def __init__(self, csv_path, split, transform_query, transform_tile):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.tq = transform_query
        self.tt = transform_tile

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # грузим обе картинки
        q = Image.open(row["query_path"]).convert("RGB")
        t = Image.open(row["tile_path"]).convert("RGB")

        # применяем свои трансформы
        q = self.tq(q)
        t = self.tt(t)

        # достаем всякую инфу
        tile_id = int(row["tile_id"])
        x = float(row["x"])
        y = float(row["y"])

        return q, t, tile_id, x, y
