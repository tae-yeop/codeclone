
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

print(sys.path)
print(os.getcwd())
os.chdir('..')
print(os.getcwd())
from ..models.stylegan2 import SynthesisBlock