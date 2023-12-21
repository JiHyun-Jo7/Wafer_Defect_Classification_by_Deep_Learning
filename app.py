from PIL import Image
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import *
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5 import uic
import sys, warnings
import pandas as pd

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

form_window = uic.loadUiType('./classification.ui')[0]
class Exam(QWidget, form_window):           # 클래스 생성
    def __init__(self):
        super().__init__()                  # 부모 클래스
        self.setupUi(self)
        super().__init__()                  # 부모 클래스
        self.setFixedWidth(450)
        self.setFixedHeight(700)
        self.setupUi(self)
        self.btn_open.clicked.connect(self.btn_open_clicked_slot)
        model_path = './models/CNN_0.949.h5'
        self.model = load_model(model_path)
        self.path = ('./Img_sample/gui.png', '')
        pixmap = QPixmap(self.path[0])
        self.lb_img.setPixmap(pixmap)

    def btn_open_clicked_slot(self):             # 버튼을 누르는 동안 반응하는 함수    (눌렀다 떼면 반응 -> released)
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self, 'Open file',
                                                './Img_sample', 'Image Files(*.jpg;*png);; All Files(*.*)')
        # 이미지 불러오기 (확장자 제한 : jpg, png, 모든 파일)
        if self.path[0] == '':
            self.path = old_path            # 불러오기를 취소할 경우 기존 사진으로 복원 (프로그램 꺼짐 방지)
        print(self.path)
        pixmap = QPixmap(self.path[0])      # 이미지 라벨을 선택한 사진으로 교체
        self.lb_img.setPixmap(pixmap)

        try:
            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((37, 37))
            data = np.asarray(img)
            data = data / 255
            print(data)
            data = data.reshape(-1, 37, 37, 1)

            pred = self.model.predict(data)
            predicted_prob = np.max(pred)
            pred_str = f'Probability: {predicted_prob * 100:.2f}%'
            self.lb_pred.setText(pred_str)

            labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
            predicted_label = labels[np.argmax(pred)]
            self.lb_label.setText(f'Predicted Label: {predicted_label}')

        except:
            print('error : {}'.format(self.path[0]))



if __name__ == '__main__':                  # 메인
    app = QApplication(sys.argv)            # 객체 생성
    MainWindow = Exam()
    MainWindow.show()                       # 창를 화면에 출력
    sys.exit(app.exec_())                   # 창 화면 출력을 유지하는 함수