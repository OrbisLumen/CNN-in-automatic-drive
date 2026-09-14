import unittest
from .check_all import TestLesson3
suite=unittest.TestSuite(TestLesson3(n) for n in ("test_manual_seed","test_chain","test_gradcheck"))
result=unittest.TextTestRunner().run(suite)
raise SystemExit(not result.wasSuccessful())
