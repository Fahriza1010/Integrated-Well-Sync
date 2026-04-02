import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)

window = QMainWindow()
window.setWindowTitle("Contoh PyQt App")
window.resize(400, 300)
window.show()

sys.exit(app.exec_())