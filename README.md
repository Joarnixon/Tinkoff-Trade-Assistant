## Торговый ассистент для биржи акций на основе ИИ и машинного обучения. Сканер по всем акциям сразу на предмет потенциально лучших точек входа.
## Данные обрабатываются по полному циклу: preprocessing, validation, feature engineering, feature selection, postprocessing и далее обучается ансамбль моделей.
## Метрики качества моделей в режиме онлайн торговли записываются на tensorboard, все операции с данными, моделями и tinkoff api происходят асинхронно.
## Доступ к предсказаниям реализован через клиентское desktop приложение по websockets с применением fastapi.

### Stack: torch, scikit-learn, tensorboard, fastapi, polars, tinkoff-investments, asyncio, hydra, OmegaConf, numpy, numba
