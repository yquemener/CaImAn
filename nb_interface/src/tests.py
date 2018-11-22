import unittest

'''
This file will include test functions to allow automated testing
of the interface. It is under development. It is difficult to automate
full testing of the notebook but at least many of the functions can be tested.

'''


class TestMotionCorrection(unittest.TestCase):

	'''    def test_upper(self):
		self.assertEqual('foo'.upper(), 'FOO')

	def test_isupper(self):
		self.assertTrue('FOO'.isupper())
		self.assertFalse('Foo'.isupper())

	def test_split(self):
		s = 'hello world'
		self.assertEqual(s.split(), ['hello', 'world'])
		# check that s.split fails when the separator is not a string
		with self.assertRaises(TypeError):
			s.split(2)'''
	pass

class TestCNMF(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()
