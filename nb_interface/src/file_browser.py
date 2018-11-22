import traitlets
from ipywidgets import widgets
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore

'''
This file includes the class and methods for launching a PyQt5 file dialog
that is used for selecting files in the interface.
'''

class FileBrowserBtn(widgets.Button):
    """A file widget that leverages PyQt5."""

    def __init__(self, desc='Browse'):
        super(FileBrowserBtn, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = desc
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
            filepath = QFileDialog.getOpenFileName()
            b.files = [filepath[0]]

class DirBrowserBtn(widgets.Button):
    """A file widget that leverages PyQt5"""

    def __init__(self, desc='Browse'):
        super(DirBrowserBtn, self).__init__()
        # Add the selected_files trait
        self.add_traits(dir=traitlets.traitlets.List())
        # Create the button.
        self.description = desc
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
            dir_ = QFileDialog.getExistingDirectory()
            b.dir = [dir_]

# my_button = SelectFilesButton()
# my_button
#
