# create a unittest stub
import unittest


class Test(unittest.TestCase):

    #set up the test
    def setUp(self):
        import os
        import pickle

        print (os.getcwd())

        with open("tests/price_mat.pkl", "rb") as f:
            self.mat, _, _ = pickle.load(f)

    def test_create_control(self):
        from robust_control import get_control
        treated_i = 0
        eta_n = 10
        mu_n = 3

        control, orig, v = get_control(self.mat, treated_i, eta_n, mu_n=mu_n, cuda=False, 
            parts=False, preint=90, train=0.79)
        


if __name__ == '__main__':
    unittest.main()