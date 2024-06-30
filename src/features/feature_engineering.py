import polars as pl

class BuildFeatures:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = [getattr(self, feature) for feature in cfg.features]

    def buy_sell_quantity_ratio(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns([
            pl.when(pl.col("sellers_quantity") != 0)
            .then(pl.col("buyers_quantity") / pl.col("sellers_quantity"))
            .otherwise(pl.col("buyers_quantity"))
            .alias("buy_sell_quantity_ratio")
        ])
        return data
    
    def slope(self, data: pl.DataFrame) -> pl.DataFrame:
        data = data.with_columns([
            ((pl.col('price_last').shift(-1, fill_value=0) - pl.col('price_last')) / 
             (pl.col('time').shift(-1, fill_value=0) - pl.col('time'))).alias('slope').shift(1, fill_value=0)
        ])

        data = data.with_columns([
            pl.when((pl.col('slope') > -1e-3) & (pl.col('slope') < 1e-3)).then(0)
            .when(pl.col('slope') > 100).then(100)
            .when(pl.col('slope') < -100).then(-100)
            .otherwise(pl.col('slope')).alias('slope')
        ])
        return data

    
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        for method in self.pipeline:
            data = method(data)
        return data