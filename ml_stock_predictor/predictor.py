from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class StockPredictor:
    slope: float = field(init=False, default=0.0)
    intercept: float = field(init=False, default=0.0)
    _is_fitted: bool = field(init=False, default=False)
    _training_points: int = field(init=False, default=0)

    def fit(self, prices: list[float]) -> "StockPredictor":
        if len(prices) < 2:
            raise ValueError("at least two prices are required to train the predictor")

        x_values = list(range(len(prices)))
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(prices) / len(prices)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, prices))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        self.slope = numerator / denominator if denominator else 0.0
        self.intercept = y_mean - (self.slope * x_mean)
        self._training_points = len(prices)
        self._is_fitted = True
        return self

    def predict_next(self, days_ahead: int = 1) -> float:
        if not self._is_fitted:
            raise ValueError("predictor must be fitted before predicting")
        if days_ahead < 1:
            raise ValueError("days_ahead must be at least 1")

        next_index = self._training_points + (days_ahead - 1)
        return self.intercept + (self.slope * next_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a future stock closing price.")
    parser.add_argument("prices", nargs="+", type=float, help="Historical closing prices.")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=1,
        help="Number of trading days ahead to predict.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictor = StockPredictor().fit(args.prices)
    prediction = predictor.predict_next(days_ahead=args.days_ahead)
    print(f"Predicted closing price in {args.days_ahead} day(s): {prediction:.2f}")
    return 0
