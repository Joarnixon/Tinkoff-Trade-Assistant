import asyncio
from .observer import Observer 
from .data_manager import DataManager, NullDataManager
from typing import Optional
import time
import yaml
from dataclasses import dataclass
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

@dataclass
class Data:
    figi: str
    bids: float = 0
    asks: float = 0
    buyers_count: int = 0
    buyers_quantity: int = 0
    weighted_bid: float = 0
    weighted_ask: float = 0
    sellers_count: int = 0
    sellers_quantity: int = 0
    price: float = 0
    last_price: float = 0

class Subject:
    def __init__(self):
        self._observers: list[Observer] = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def remove(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, data: dict):
        for observer in self._observers:
            observer.update(data)

class DataBuffer(Data):
    """
    Initialize a new instance of DataBuffer for a specific share.

    Parameters:
    figi (str)
    data_manager (DataManager, optional): An instance of DataManager to handle data storage.
    Defaults to NullDataManager if not provided and will not save anything.
    """
    def __init__(self, figi: str, data_manager: Optional[DataManager], buffer_write_delay):
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
        self.bids = 0
        self.asks = 0
        self.weighted_ask = 0
        self.weighted_bid = 0

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
    def __init__(self, cfg, data_manager=None):
        super().__init__()
        self.cfg = cfg
        self.data_manager = data_manager if data_manager is not None else NullDataManager()

        with open(cfg.paths.shares_dict, 'r', encoding='utf-8') as file:
            stocks_figi = list(yaml.safe_load(file).keys()) # only figi needed

        self.shares_dict = {figi: DataBuffer(figi, self.data_manager, cfg.data.data_gather.buffer_write_delay) for figi in stocks_figi}

        self.orderbook_depth = cfg.data.data_gather.orderbook_depth
        self.request_delay = cfg.data.data_gather.request_delay
        self.single_data_write_delay = cfg.data.data_gather.single_data_write_delay


    @staticmethod
    def _update_share_data_trade(share: DataBuffer, trade):
        share.last_price = qtd(trade.price)
        if trade.direction == TradeDirection.TRADE_DIRECTION_BUY:
            share.buyers_count += 1
            share.buyers_quantity += trade.quantity
        elif trade.direction == TradeDirection.TRADE_DIRECTION_SELL:
            share.sellers_count += 1
            share.sellers_quantity += trade.quantity

    @staticmethod
    def _update_share_data_orderbook(share: DataBuffer, orderbook):
        share.bids = sum([bid.quantity for bid in orderbook.bids])
        share.asks = sum([ask.quantity for ask in orderbook.asks])
        share.weighted_bid = round(sum([bid.quantity * qtd(bid.price) for bid in orderbook.bids]) / share.bids, 5)
        share.weighted_ask = round(sum([ask.quantity * qtd(ask.price) for ask in orderbook.asks]) / share.asks, 5)
        share.price = share.last_price

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
            async for marketdata in client.market_data_stream.market_data_stream(request_iterator()):
                if marketdata.trade is not None:
                    share: DataBuffer = self.shares_dict[marketdata.trade.figi]

                    self._update_share_data_trade(share, marketdata.trade)
                    if time.time() - share.last_write_time > self.single_data_write_delay: 
                        share.price = qtd(marketdata.trade.price)
                        data_point = list(map(float, [share.buyers_count, share.buyers_quantity, share.sellers_count,
                                    share.sellers_quantity, share.price, share.bids,
                                    share.asks, share.weighted_bid, share.weighted_ask]))
                        share.append(data_point)
                        self.notify({marketdata.trade.figi: data_point})  # Notify listeners with new data
                        share.reset_data()
                        share.last_write_time = time.time()

                if marketdata.orderbook is not None:
                    share = self.shares_dict[marketdata.orderbook.figi]
                    self._update_share_data_orderbook(share, marketdata.orderbook)

    async def run(self):
        await self.collect()