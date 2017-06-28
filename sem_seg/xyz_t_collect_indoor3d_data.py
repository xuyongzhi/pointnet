import os
import sys

print(__file__)
BASE_DIR = os.path.dirname( os.path.abspath(__file__) )
print(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
sys.path.append(BASE_DIR)

import indoor3d_util


