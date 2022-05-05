import unittest

from project import download_dataset


class MyTestCase(unittest.TestCase):
    def test_split(self):
        train_ds, val_ds, test_ds = download_dataset()
        pass


if __name__ == '__main__':
    unittest.main()
