import polars as pl

class BuildFeatures:
    def __init__(self, cfg, features: list[str]):
        self.cfg = cfg
        self.pipeline = [getattr(self, feature) for feature in features]

    def buy_sell_quantity_ratio(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns([
            pl.when(pl.col("sellers_quantity") != 0)
            .then(pl.col("buyers_quantity") / pl.col("sellers_quantity"))
            .otherwise(pl.col("buyers_quantity"))
            .alias("buy_sell_quantity_ratio")
        ])
        return data
    
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        for method in self.pipeline:
            data = method(data)
        return data