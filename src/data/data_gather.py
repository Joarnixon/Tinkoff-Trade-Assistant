import asyncio
from numpy import nan
from .subject import Subject
from .data_manager import DataManager, NullDataManager
from .data_preprocess import OrderBookPreprocessor
from typing import Optional, Union
import time
import yaml
from dataclasses import dataclass, field
import polars as pl
from tinkoff.invest.utils import now, quotation_to_decimal as qtd
from tinkoff.invest import (
    AsyncClient,
    Client,
    MarketDataRequest,
    SubscribeOrderBookRequest,
    SubscribeTradesRequest,
    SubscriptionAction,
    OrderBookInstrument,
    TradeInstrument,
    TradeDirection
)
from tinkoff.invest.exceptions import AioRequestError

@dataclass
class Data:
    figi: str
    buyers_count: int = 0
    buyers_quantity: int = 0
    sellers_count: int = 0
    sellers_quantity: int = 0
    last_price: float = 0
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)
    weighted_bid: dict = field(default_factory=dict)
    weighted_ask: dict = field(default_factory=dict)
    price: dict = field(default_factory=dict)
    bid_to_ask_ratio: dict = field(default_factory=dict)

    @property
    def time(self):
        return int(time.time())

class DataBuffer(Data):
    """
    Initialize a new instance of DataBuffer for a specific share.

    Parameters:
    figi (str)
    data_manager (DataManager, optional): An instance of DataManager to handle data storage.
    Defaults to NullDataManager if not provided and will not save anything.
    """
    def __init__(self, figi: str, data_manager: Union[DataManager, NullDataManager], buffer_write_delay):
        super().__init__(figi)
        self.data_buffer = []
        self.buffer_write_delay = buffer_write_delay
        self.last_write_time = time.time()
        self.last_buffer_write_time = time.time()
        self.data_manager = data_manager

    def reset_data(self):
        self.buyers_count = 0
        self.buyers_quantity = 0
        self.sellers_count = 0
        self.sellers_quantity = 0
        self.price = {}
        self.bids = {}
        self.asks = {}
        self.weighted_bid = {}
        self.weighted_ask = {}
        self.bid_to_ask_ratio = {}
        # not last_price

    def append(self, data_point):
        self.data_buffer.append(data_point)

        if time.time() - self.last_buffer_write_time > self.buffer_write_delay:  # Write every $buffer_write_time seconds
            self._write_data_buffer()

    def _write_data_buffer(self):
        self.data_manager.write_share(self.figi, self.data_buffer)
        self.data_buffer = []
        self.last_buffer_write_time = time.time()

class DataCollector(Subject):
    """
    Usage: Pass config file, call collect(). 
    To listen to incoming data, call DataCollector.attach(listener).

    Parameters:
    cfg (Config): A configuration object containing necessary parameters.
    data_manager (DataManager, optional): An instance of DataManager to handle data storage.
    """
    def __init__(self, cfg, data_manager: Optional[DataManager] = None):
        super().__init__()
        self.cfg = cfg
        self.data_manager = data_manager if data_manager is not None else NullDataManager()
        self.orderbook_preprocessor = OrderBookPreprocessor(processes=self.cfg.data.data_gather.orderbook_processes)

        with open(cfg.paths.shares_dict, 'r', encoding='utf-8') as file:
            stocks_figi = list(yaml.safe_load(file).keys()) # only figi needed

        self.shares_dict = {figi: DataBuffer(figi, self.data_manager, cfg.data.data_gather.buffer_write_delay) for figi in stocks_figi}

        self.orderbook_depth = cfg.data.data_gather.orderbook_depth
        self.request_delay = cfg.data.data_gather.request_delay
        self.single_data_write_delay = cfg.data.data_gather.single_data_write_delay

        # set initial price
        with Client(cfg.tokens.TOKEN) as client:
            for share in list(self.shares_dict.values()):
                share.last_price = qtd(client.market_data.get_last_prices(figi=[share.figi]).last_prices[0].price)

    # Modify this to take more data if you know how. Change columns in config
    @staticmethod
    def _update_share_data_trade(share: DataBuffer, trade):
        share.last_price = qtd(trade.price)
        if trade.direction == TradeDirection.TRADE_DIRECTION_BUY:
            share.buyers_count += 1
            share.buyers_quantity += trade.quantity
        elif trade.direction == TradeDirection.TRADE_DIRECTION_SELL:
            share.sellers_count += 1
            share.sellers_quantity += trade.quantity

    # Modify this to take more data if you know how. Change columns in config
    @staticmethod
    def _update_share_data_orderbook(share: DataBuffer, orderbook):
        timestamp = int(time.time())
        try:
            share.price[timestamp] = share.last_price
            share.bids[timestamp] = sum([bid.quantity for bid in orderbook.bids])
            share.asks[timestamp] = sum([ask.quantity for ask in orderbook.asks])
            share.weighted_bid[timestamp] = round(sum([bid.quantity * qtd(bid.price) for bid in orderbook.bids]) / share.bids[timestamp], 5)
            share.weighted_ask[timestamp] = round(sum([ask.quantity * qtd(ask.price) for ask in orderbook.asks]) / share.asks[timestamp], 5)
            share.bid_to_ask_ratio[timestamp] = round(share.bids[timestamp] / share.asks[timestamp], 5)
        except ZeroDivisionError:
            share.price[timestamp] = nan
            share.bids[timestamp] = nan
            share.asks[timestamp] = nan
            share.weighted_bid[timestamp] = nan
            share.weighted_ask[timestamp] = nan
            share.bid_to_ask_ratio[timestamp] = nan
    
    def make_datapoint(self, share: DataBuffer, trade):
        share.last_price = qtd(trade.price)

        orderbook_features = self.orderbook_preprocessor.transform([getattr(share, column) for column in self.cfg.data.raw_data_orderbook_columns])
        data_point = list(map(float, [getattr(share, column) for column in self.cfg.data.raw_data_trades_columns] + orderbook_features))
        share.append(data_point)
        self.notify({trade.figi: [data_point]})  # Notify listeners with new data
        share.reset_data()
        share.last_write_time = time.time()

    async def collect(self):
        async def request_iterator():
            shares = self.shares_dict
            delay = self.request_delay
            depth = self.orderbook_depth
            yield MarketDataRequest(
                subscribe_trades_request=SubscribeTradesRequest(
                    subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[TradeInstrument(figi=figi) for figi in shares])
            )
            yield MarketDataRequest(
                subscribe_order_book_request=SubscribeOrderBookRequest(
                    subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                    instruments=[OrderBookInstrument(figi=figi, depth=depth) for figi in shares])
            )
            while True:
                await asyncio.sleep(delay)

        async with AsyncClient(self.cfg.tokens.TOKEN) as client:
            while True:
                try:
                    async for marketdata in client.market_data_stream.market_data_stream(request_iterator()):
                        if marketdata.trade is not None:
                            share: DataBuffer = self.shares_dict[marketdata.trade.figi]

                            self._update_share_data_trade(share, marketdata.trade)
                            if time.time() - share.last_write_time > self.single_data_write_delay: 
                                self.make_datapoint(share, marketdata.trade)

                        if marketdata.orderbook is not None:
                            share = self.shares_dict[marketdata.orderbook.figi]
                            self._update_share_data_orderbook(share, marketdata.orderbook)

                except AioRequestError as e:
                    metadata = e.metadata
                    ratelimit_reset = int(metadata.ratelimit_reset) + 1
                    print(f"Rate limit exceeded. Waiting {ratelimit_reset:.2f} seconds...")
                    await asyncio.sleep(ratelimit_reset)
                

    async def run(self):
        await self.collect()