# create a unittest stub
import unittest
import torch
import numpy as np


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

    def test_prepare_data(self):
        from robust_control import prepare_data, _make_params, prepare_data_b, _make_params_b
        treated_i = 0
        eta_n = 10
        mu_n = 3
        preint=90
        parts=False
        rows = [0,1]

        etas, mus, cutoff, parts, denoise = _make_params(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     parts=parts,
                                                     treated_i=treated_i)
        etasb, musb, cutoffb, partsb, denoiseb = _make_params_b(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     rows=rows,
                                                     parts=parts,
                                                     )
        
        Y1_ob, Y0_ob, etasb, ab, bb = prepare_data_b(
                torch.Tensor(self.mat), rows, etasb, musb, denoise=denoiseb)
        
        Y1_o, Y0_o, etas, a, b = prepare_data(self.mat, treated_i, etas, mus, denoise=denoise)

        # check that Y1_ob and Y0_ob's first dimension have size 2
        self.assertEqual(Y1_ob.shape[0], 2)
        self.assertEqual(Y0_ob.shape[0], 2)

        # check that Y1_ob[0] and Y1_o are the same
        self.assertEqual(Y1_ob[0].shape, Y1_o.shape)
        self.assertEqual(Y1_ob.type(), Y1_o.type())
        self.assertTrue(torch.allclose(Y1_ob[0], Y1_o))

        # check Y0_ob[0] and Y0_o are the same
        self.assertEqual(Y0_ob[0].shape, Y0_o.shape)
        self.assertEqual(Y0_ob.type(), Y0_o.type())
        self.assertTrue(torch.allclose(Y0_ob[0], Y0_o, atol=1e-6))

        # check that etasb[0] and etas are the same
        self.assertEqual(etasb[0].shape, etas.shape)
        print (etasb[0].type(), etas.type())
        self.assertTrue(torch.allclose(etasb[0], etas))
        # check first dimension
        self.assertEqual(etasb.shape[0], 2)

        # check that ab[0] and a are the same
        self.assertEqual(ab, a)
        self.assertEqual(bb, b)

    def test_get_training_data(self):
        
        from robust_control import prepare_data, _make_params, prepare_data_b, _make_params_b
        from robust_control import _get_train_data
        treated_i = 0
        eta_n = 10
        mu_n = 3
        preint=90
        parts=False
        rows = [0,1]
        train=0.8

        etas, mus, cutoff, parts, denoise = _make_params(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     parts=parts,
                                                     treated_i=treated_i)
        etasb, musb, cutoffb, partsb, denoiseb = _make_params_b(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     rows=rows,
                                                     parts=parts,
                                                     )
        
        Y1_ob, Y0_ob, etasb, ab, bb = prepare_data_b(
                torch.Tensor(self.mat), rows, etasb, musb, denoise=denoiseb)
        
        Y1_o, Y0_o, etas, a, b = prepare_data(self.mat, treated_i, etas, mus, denoise=denoise)

        train_i = int(np.floor(train*cutoff))

        Y1_t, Y0_t = _get_train_data(Y1_o, Y0_o, train_i, parts)

        Y1_tb, Y0_tb = _get_train_data(Y1_ob, Y0_ob, train_i, parts)

        # check that Y1_tb and Y0_tb's first dimension have size 2
        self.assertEqual(Y1_tb.shape[0], 2)
        self.assertEqual(Y0_tb.shape[0], 2)

        # check that Y1_tb[0] and Y1_t are the same
        self.assertEqual(Y1_tb[0].shape, Y1_t.shape)
        self.assertEqual(Y1_tb.type(), Y1_t.type())
        self.assertTrue(torch.allclose(Y1_tb[0], Y1_t))

        # check Y0_tb[0] and Y0_t are the same
        self.assertEqual(Y0_tb[0].shape, Y0_t.shape)
        self.assertEqual(Y0_tb.type(), Y0_t.type())
        self.assertTrue(torch.allclose(Y0_tb[0], Y0_t, atol=1e-6))

    # a method for estimate_weights function
    def test_estimate_weights(self):
        from robust_control import prepare_data, _make_params, prepare_data_b, _make_params_b
        from robust_control import _get_train_data, estimate_weights_b, estimate_weights_bb
        treated_i = 0
        eta_n = 10
        mu_n = 3
        preint=90
        parts=False
        rows = [0,1]
        train=0.8

        etas, mus, cutoff, parts, denoise = _make_params(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     parts=parts,
                                                     treated_i=treated_i)
        etasb, musb, cutoffb, partsb, denoiseb = _make_params_b(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     rows=rows,
                                                     parts=parts,
                                                     )
        
        Y1_ob, Y0_ob, etasb, ab, bb = prepare_data_b(
                torch.Tensor(self.mat), rows, etasb, musb, denoise=denoiseb)
        
        Y1_o, Y0_o, etas, a, b = prepare_data(self.mat, treated_i, etas, mus, denoise=denoise)

        train_i = int(np.floor(train*cutoff))

        Y1_t, Y0_t = _get_train_data(Y1_o, Y0_o, train_i, parts)

        Y1_tb, Y0_tb = _get_train_data(Y1_ob, Y0_ob, train_i, parts)

        vs = estimate_weights_b(Y1_t, 
                                Y0_t, 
                                etas)

        vsb = estimate_weights_bb(Y1_tb, 
                                  Y0_tb, 
                                  etasb
                                  )

        # check that vsb and vs's first dimension have size 2
        self.assertEqual(vsb.shape[0], 2)

        # check that vsb[0] and vs are the same
        self.assertEqual(vsb[0].shape, vs.shape)
        self.assertEqual(vsb.type(), vs.type())
        
        # The SVD approximation algorithm is not deterministic,
        # so we use a loose tolerance here.
        self.assertTrue(torch.allclose(vsb[0][0], vs[0], atol=1e-3))


    # next test method
    def test_estimate_weights2(self):
        # a method for estimate_weights function
        from robust_control import prepare_data, _make_params, prepare_data_b, _make_params_b
        from robust_control import _get_train_data, estimate_weights_b, estimate_weights_bb
        from robust_control import loss_fn, unbind_data
        treated_i = 0
        eta_n = 10
        mu_n = 3
        preint=90
        parts=False
        rows = [0,1]
        train=0.8

        etas, mus, cutoff, parts, denoise = _make_params(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     parts=parts,
                                                     treated_i=treated_i)
        etasb, musb, cutoffb, partsb, denoiseb = _make_params_b(self.mat, 
                                                     eta_n, 
                                                     mu_n, 
                                                     preint=preint, 
                                                     rows=rows,
                                                     parts=parts,
                                                     )
        
        Y1_ob, Y0_ob, etasb, ab, bb = prepare_data_b(
                torch.Tensor(self.mat), rows, etasb, musb, denoise=denoiseb)
        
        Y1_o, Y0_o, etas, a, b = prepare_data(self.mat, treated_i, etas, mus, denoise=denoise)

        train_i = int(np.floor(train*cutoff))

        Y1_t, Y0_t = _get_train_data(Y1_o, Y0_o, train_i, parts)

        Y1_tb, Y0_tb = _get_train_data(Y1_ob, Y0_ob, train_i, parts)

        vs = estimate_weights_b(Y1_t, 
                                Y0_t, 
                                etas)

        vsb = estimate_weights_bb(Y1_tb, 
                                  Y0_tb, 
                                  etasb
                                  )
        
        Y1_eta = vs.mT @ Y0_o

        Y1_etab = vsb.mT @ Y0_ob

        self.assertTrue(torch.allclose(Y1_eta, Y1_etab[0], atol=1e-4))

        min_idx = loss_fn(Y1_o[:, :, train_i:cutoff], Y1_eta[:, :, train_i:cutoff]).argmin()

        loss = loss_fn(Y1_ob[:, :, :, train_i:cutoff], Y1_etab[:, :, :, train_i:cutoff])
        min_idx_b = loss.argmin(dim=1)

        self.assertEqual(min_idx,  min_idx_b[0][0])

        Y1_n, Y0_n = _get_train_data(Y1_o[min_idx, :, :].unsqueeze(0), 
                                 Y0_o[min_idx, :, :].unsqueeze(0), 
                                 cutoff, parts)
        
        Y1_nb, Y0_nb = _get_train_data(Y1_ob[:, min_idx_b[:,0]], 
                                 Y0_ob[:, min_idx_b[:,0]], 
                                 cutoff, parts)
        
        self.assertTrue(torch.allclose(Y1_nb[0][0], Y1_n[0], atol=1e-6))

        vs = estimate_weights_b(Y1_n, Y0_n, etas[min_idx].unsqueeze(0))

        vsb = estimate_weights_bb(Y1_nb, Y0_nb, etasb[:, min_idx_b[:,0]])

        self.assertTrue(torch.allclose(vsb[0][0], vs[0], atol=1e-3))


        Y1_hats = unbind_data(vs.mT @ Y0_o, a, b)
        Y1s = unbind_data(Y1_o, a, b)

        Y1_hatsb = unbind_data(vsb.mT @ Y0_ob[:, min_idx_b[:,0]], ab, bb)
        Y1sb = unbind_data(Y1_ob[:, min_idx_b[:,0]], ab, bb)

        rat = Y1_hats[0]/Y1_hatsb[0][0]

        self.assertTrue(torch.allclose(rat, torch.ones_like(Y1_hats[0]), atol=1e-2))

        rat = Y1s[0]/Y1sb[0][0]

        self.assertTrue(torch.allclose(rat, torch.ones_like(Y1s[0]), atol=1e-3))
        
    # make a method to test the get_control/get_controls top to bottom
    def test_get_control(self):
        from robust_control import get_control, get_controls
        
        import pickle
        with open("tests/price_mat.pkl", "rb") as f:
            price_mat, _, _ = pickle.load(f)

        data = price_mat[:, -35:]
        
        # prepare arguments for get_control
        treated_i = 0
        eta_n = 10
        mu_n = 3
        preint=90
        parts=False
        rows = [0,1]
        train=0.8

        # call get_control()
        control, fact, v = get_control(self.mat, 
                                        treated_i, 
                                        eta_n, 
                                        mu_n, 
                                        preint=preint, 
                                        parts=parts, 
                                        train=train)
        
        # call get_controls()
        controls_b, facts_b, vs_b = get_controls(data, [0], mu_n=3)
        
        #print (v.flatten()/vs_b[0].flatten())
        
    # make a method to test bind_data_b
    def test_bind_data_b(self):
        # import the function to test
        from robust_control import bind_data_b

        


if __name__ == '__main__':
    unittest.main()