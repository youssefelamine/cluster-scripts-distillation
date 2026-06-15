import os
import sys


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REINFORCEMENT_DIR = os.path.join(REPOSITORY_ROOT, "reinforcement")
if REINFORCEMENT_DIR not in sys.path:
    sys.path.insert(0, REINFORCEMENT_DIR)
