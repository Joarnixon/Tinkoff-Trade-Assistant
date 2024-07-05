from src.data.observer import Observer

import asyncio

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, data):
        for observer in self._observers:
            if asyncio.iscoroutinefunction(observer.update):
                asyncio.create_task(observer.update(data))
            else:
                observer.update(data)