# create a stub for unittest
import unittest

import numpy as np
import torch

# import the module to be tested
import robust_control as rc

# create a class to test the class to be tested
class TestRobustControl(unittest.TestCase):
    
        # create a test method
        def test_make_params(self):
            # create a (10, 20) numpy array with random values
            mat = np.random.rand(10, 20)

            # set sensible values for the arguments of
            # `rc._make_params()`, using `mat` as the first
            # argument
            treated_i = 1
            eta_n = 10
            mu_n = 3
            preint = 15
            parts = None

            # call the function to be tested
            etas, mus, cutoff, parts, denoise = rc._make_params(
                 mat, 
                 eta_n, 
                 mu_n, 
                 preint, 
                 parts=parts,
                 treated_i=treated_i)
            
            # check that the function returns the expected values
            # assert that `etas` is of type List
            self.assertIsInstance(etas, list)
            # assert that `mus`` is of type torch.Tensor
            self.assertIsInstance(mus, torch.Tensor)

            # assert that `mus` has the correct shape
            self.assertEqual(mus.shape, (mu_n, 1))

            # assert that `etas` has length `eta_n`
            self.assertEqual(len(etas), eta_n)
            # assert that `mus` has length `mu_n`
            self.assertEqual(len(mus), mu_n)

            # assert that `cutoff` is equal to the number of columns in
            # `mat` minus `preint`
            self.assertEqual(cutoff, mat.shape[1] - preint)

            # assert that `parts` is equal to 0
            self.assertEqual(parts, 0)

            # assert that `denoise` is True
            self.assertTrue(denoise)

        def test_prepare_data(self):
             # test the function `rc.prepare_data()`
            # create a (10, 20) numpy array with random values
            mat = np.random.rand(10, 20)

            # make parameters, using sensible
            # values for the arguments of `rc._make_params()`
            treated_i = 1
            eta_n = 10
            mu_n = None
            preint = 15
            parts = None
            etas, mus, cutoff, parts, denoise = rc._make_params(
                 mat, 
                 eta_n, 
                 mu_n, 
                 preint, 
                 parts=parts,
                 treated_i=treated_i)
            
            # call the function to be tested
            data = rc.prepare_data(mat, treated_i, etas, mus, denoise)



# run the tests
if __name__ == '__main__':
    unittest.main()