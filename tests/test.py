# create a unittest stub
import unittest


class Test(unittest.TestCase):

    #set up the test
    def setUp(self):
        import os
        import pickle

        with open("tests/price_mat.pkl", "rb") as f:
            self.mat, _, _ = pickle.load(f)

    def test_create_control(self):
        from robust_control import get_control
        treated_i = 0
        eta_n = 10
        mu_n = 3

        control, orig, v = get_control(self.mat, treated_i, eta_n, mu_n=mu_n, cuda=False, 
            parts=10, preint=90, train=0.79)
        
        self.assertEqual(control.shape, orig.shape)
        self.assertEqual(control.shape[0], 1)
        self.assertEqual(control.shape[1], 121)
        self.assertEqual(v.shape[0], 299)
        self.assertEqual(v.shape[1], 1)



if __name__ == '__main__':
    unittest.main()