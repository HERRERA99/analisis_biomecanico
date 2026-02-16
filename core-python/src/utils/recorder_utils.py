import pandas as pd


class DataRecorder:
    def __init__(self):
        self.rows = []

    def add(self, row: dict):
        self.rows.append(row)

    def save(self, path):
        df = pd.DataFrame(self.rows)
        df.to_parquet(path)
