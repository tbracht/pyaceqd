# import everything for unittests
import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyaceqd.tools import output_ops_dm, compose_dm

# test outputs of pyaceqd.tools.output_ops_dm
class TestOutputOpsDM(unittest.TestCase):
    def test_output_ops_dm(self):
        operators = output_ops_dm(2)
        self.assertEqual(len(operators), 3)
        operators_expected = ["|0><0|_2", "|0><1|_2", "|1><1|_2"]
        for i in range(len(operators)):
            self.assertEqual(operators[i], operators_expected[i])
        
        operators = output_ops_dm(6)
        self.assertEqual(len(operators), int((6*6-6)/2)+6)
        operators_expected = ['|0><0|_6', '|0><1|_6', '|0><2|_6', '|0><3|_6', '|0><4|_6', '|0><5|_6', '|1><1|_6', '|1><2|_6', 
                              '|1><3|_6', '|1><4|_6', '|1><5|_6', '|2><2|_6', '|2><3|_6', 
                              '|2><4|_6', '|2><5|_6', '|3><3|_6', '|3><4|_6', '|3><5|_6', '|4><4|_6', '|4><5|_6', '|5><5|_6']
        for i in range(len(operators)):
            self.assertEqual(operators[i], operators_expected[i])

    def test_compose_dm(self):
        dim = 2
        rho = np.zeros((1, dim,dim),dtype=np.complex128)
        rho[0,0,0] = 1
        rho[0,1,1] = 2
        rho[0,0,1] = 3+3j
        rho[0,1,0] = 3-3j
        # fill array that is transformed to density matrix
        data = np.zeros((4,1), dtype=complex)
        data[0,0] = 0
        data[1,0] = 1
        data[2,0] = 3+3j
        data[3,0] = 2
        t, rho_test = compose_dm(data, dim)
        self.assertTrue(np.allclose(rho, rho_test))
        self.assertTrue(np.allclose(t, np.array([0])))

    def test_output_ops_cavity(self):
        operators_21 = output_ops_dm((2,1))
        self.assertEqual(len(operators_21), 3)
        operators_21_expected = ['|0><0|_2 otimes |0><0|_1', '|0><1|_2 otimes |0><0|_1', '|1><1|_2 otimes |0><0|_1']
        for i in range(len(operators_21)):
            self.assertEqual(operators_21[i], operators_21_expected[i])
        
        dim = [2,2]
        operators_22 = output_ops_dm(dim)
        num_ops = int((np.prod(dim)**2-np.prod(dim))/2)+np.prod(dim)
        self.assertEqual(len(operators_22), num_ops)
        operators_22_expected = ['|0><0|_2 otimes |0><0|_2', '|0><0|_2 otimes |0><1|_2', '|0><1|_2 otimes |0><0|_2', '|0><1|_2 otimes |0><1|_2', '|0><0|_2 otimes |1><1|_2', '|0><1|_2 otimes |1><0|_2', 
                                 '|0><1|_2 otimes |1><1|_2', '|1><1|_2 otimes |0><0|_2', '|1><1|_2 otimes |0><1|_2', '|1><1|_2 otimes |1><1|_2']
        for i in range(len(operators_22)):
            self.assertEqual(operators_22[i], operators_22_expected[i])

        dim = [2,2,2]
        operators_222 = output_ops_dm(dim)
        num_ops = int((np.prod(dim)**2-np.prod(dim))/2)+np.prod(dim)
        self.assertEqual(len(operators_222), num_ops)
        operators_222_expected = ['|0><0|_2 otimes |0><0|_2 otimes |0><0|_2', '|0><0|_2 otimes |0><0|_2 otimes |0><1|_2', '|0><0|_2 otimes |0><1|_2 otimes |0><0|_2', '|0><0|_2 otimes |0><1|_2 otimes |0><1|_2', 
                                  '|0><1|_2 otimes |0><0|_2 otimes |0><0|_2', '|0><1|_2 otimes |0><0|_2 otimes |0><1|_2', '|0><1|_2 otimes |0><1|_2 otimes |0><0|_2', '|0><1|_2 otimes |0><1|_2 otimes |0><1|_2', 
                                  '|0><0|_2 otimes |0><0|_2 otimes |1><1|_2', '|0><0|_2 otimes |0><1|_2 otimes |1><0|_2', '|0><0|_2 otimes |0><1|_2 otimes |1><1|_2', '|0><1|_2 otimes |0><0|_2 otimes |1><0|_2', 
                                  '|0><1|_2 otimes |0><0|_2 otimes |1><1|_2', '|0><1|_2 otimes |0><1|_2 otimes |1><0|_2', '|0><1|_2 otimes |0><1|_2 otimes |1><1|_2', '|0><0|_2 otimes |1><1|_2 otimes |0><0|_2', 
                                  '|0><0|_2 otimes |1><1|_2 otimes |0><1|_2', '|0><1|_2 otimes |1><0|_2 otimes |0><0|_2', '|0><1|_2 otimes |1><0|_2 otimes |0><1|_2', '|0><1|_2 otimes |1><1|_2 otimes |0><0|_2', 
                                  '|0><1|_2 otimes |1><1|_2 otimes |0><1|_2', '|0><0|_2 otimes |1><1|_2 otimes |1><1|_2', '|0><1|_2 otimes |1><0|_2 otimes |1><0|_2', '|0><1|_2 otimes |1><0|_2 otimes |1><1|_2', 
                                  '|0><1|_2 otimes |1><1|_2 otimes |1><0|_2', '|0><1|_2 otimes |1><1|_2 otimes |1><1|_2', '|1><1|_2 otimes |0><0|_2 otimes |0><0|_2', '|1><1|_2 otimes |0><0|_2 otimes |0><1|_2', 
                                  '|1><1|_2 otimes |0><1|_2 otimes |0><0|_2', '|1><1|_2 otimes |0><1|_2 otimes |0><1|_2', '|1><1|_2 otimes |0><0|_2 otimes |1><1|_2', '|1><1|_2 otimes |0><1|_2 otimes |1><0|_2', 
                                  '|1><1|_2 otimes |0><1|_2 otimes |1><1|_2', '|1><1|_2 otimes |1><1|_2 otimes |0><0|_2', '|1><1|_2 otimes |1><1|_2 otimes |0><1|_2', '|1><1|_2 otimes |1><1|_2 otimes |1><1|_2']
        for i in range(len(operators_222)):
            self.assertEqual(operators_222[i], operators_222_expected[i])
        
if __name__ == '__main__':
    unittest.main()
