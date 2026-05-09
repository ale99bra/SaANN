import unittest
import numpy as np
from saann.metrics import Metrics


class TestMetrics(unittest.TestCase):
    """Test suite for Metrics class"""

    def setUp(self):
        """Initialize test fixtures before each test"""
        self.data = 100
        self.classes = 5
        logits = np.random.randn(self.data, self.classes)
        exp_logits = np.exp(logits)
        self.pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.true = np.random.randn(self.data, self.classes)

    def create_dataset(self, classes, data):
        """Helper method to create a dataset for testing"""
        tmp = []
        if classes > 1:
            for yi in self.true:
                one_hot = np.zeros_like(yi)
                one_hot[np.argmax(yi)] = 1
                tmp.append(one_hot)
            self.true = np.asarray(tmp)
        else:
            pred = np.random.randn(data, classes)
            pred += abs(np.min(pred))
            pred /= np.max(pred)

            for yi in self.true:
                if yi >= 0.5:
                    tmp.append(1)
                else:
                    tmp.append(0)
            self.true = np.asarray(tmp).reshape(-1, 1)

    def test_metrics_initialization(self):
        """Test that Metrics initializes correctly"""
        metrics = Metrics(y_pred=self.y_pred, y_test=self.y_true)
        self.assertIsNotNone(metrics)

    def test_report_runs_without_error(self):
        """Test that report method runs without errors"""
        for classes in [1, 2, 5]:
            with self.subTest(classes=classes):
                self.create_dataset(classes=classes, data=self.data)
            metrics = Metrics(y_pred=self.y_pred, y_test=self.y_true)
            
            # Should not raise any exception
            try:
                metrics.report()
            except Exception as e:
                self.fail(f"report() raised {type(e).__name__} unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()