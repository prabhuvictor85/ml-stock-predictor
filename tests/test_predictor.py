import subprocess
import sys
import unittest

from ml_stock_predictor import StockPredictor


class StockPredictorTests(unittest.TestCase):
    def test_predict_next_follows_linear_trend(self) -> None:
        predictor = StockPredictor().fit([100.0, 102.0, 104.0, 106.0])

        prediction = predictor.predict_next(days_ahead=2)

        self.assertAlmostEqual(prediction, 110.0)

    def test_fit_requires_at_least_two_prices(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two prices"):
            StockPredictor().fit([100.0])

    def test_cli_outputs_prediction(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml_stock_predictor",
                "100",
                "102",
                "104",
                "106",
                "--days-ahead",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Predicted closing price in 2 day(s): 110.00", result.stdout)


if __name__ == "__main__":
    unittest.main()
