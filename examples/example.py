import logging as log
import os
import sys
from tkinter import Tk

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # noqa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from process.main import GraphicalUserInterface  # noqa

app = GraphicalUserInterface(Tk())
app.frame.mainloop()
