import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_flask_app_module():
    module_path = Path(__file__).resolve().parents[1] / "flask" / "app.py"
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("plank_flask_app_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_flask_app_module()

    def test_client_ip_ignores_forwarded_header_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            with self.module.app.test_request_context(
                "/health",
                headers={"X-Forwarded-For": "203.0.113.9, 10.0.0.1"},
                environ_base={"REMOTE_ADDR": "10.0.0.1"},
            ):
                self.assertEqual(self.module._client_ip(), "10.0.0.1")

    def test_client_ip_uses_forwarded_header_for_trusted_proxy(self):
        with patch.dict(
            os.environ,
            {
                "PLANK_TRUST_PROXY_HEADERS": "true",
                "PLANK_TRUSTED_PROXY_IPS": "10.0.0.1",
            },
            clear=False,
        ):
            with self.module.app.test_request_context(
                "/health",
                headers={"X-Forwarded-For": "203.0.113.9, 10.0.0.1"},
                environ_base={"REMOTE_ADDR": "10.0.0.1"},
            ):
                self.assertEqual(self.module._client_ip(), "203.0.113.9")


if __name__ == "__main__":
    unittest.main()
