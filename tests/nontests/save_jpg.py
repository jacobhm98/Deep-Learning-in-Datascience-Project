import unittest

from data_utils import output_jpg_dir_of_training_data


class MyTestCase(unittest.TestCase):
    # Gans comment
    def test_save_jgp(self):
        output_jpg_dir_of_training_data("./jpegs/")


if __name__ == '__main__':
    unittest.main()
