from ui import *
from mainclass import *
import sys
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    cl = MainClass(ui)

    ui.pushButton_path_1.clicked.connect(cl.chose_path_1)
    ui.pushButton_path_2.clicked.connect(cl.chose_path_2)
    ui.pushButton_path_3.clicked.connect(cl.chose_path_3)
    ui.pushButton_path_4.clicked.connect(cl.chose_path_4)
    ui.pushButton_koef.clicked.connect(cl.main_video_col)
    ui.pushButton_save.clicked.connect(cl.save_path)
    ui.pushButton_load.clicked.connect(cl.load_path)



    Dialog.show()
    sys.exit(app.exec_())