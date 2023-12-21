from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
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


class Exam(QWidget, form_window):  # 클래스 생성
    def __init__(self):
        super().__init__()  # 부모 클래스
        self.setupUi(self)
        self.setFixedWidth(450)
        self.setFixedHeight(700)
        self.btn_open.clicked.connect(self.btn_open_clicked_slot)
        self.cb.activated[str].connect(self.Changed_Str)
        self.le.textChanged[str].connect(self.Changed_Str)
        model_path = './models/CNN_0.927.h5'
        self.model = load_model(model_path)
        self.path = ('./Img_sample/gui.png', '')
        pixmap = QPixmap(self.path[0])
        self.lb_img.setPixmap(pixmap)

    def btn_open_clicked_slot(self):  # 버튼을 누르는 동안 반응하는 함수    (눌렀다 떼면 반응 -> released)
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self, 'Open file',
                                                './Img_sample', 'Image Files(*.jpg;*png);; All Files(*.*)')
        # 이미지 불러오기 (확장자 제한 : jpg, png, 모든 파일)
        if self.path[0] == '':
            self.path = old_path  # 불러오기를 취소할 경우 기존 사진으로 복원 (프로그램 꺼짐 방지)
        print(self.path)
        pixmap = QPixmap(self.path[0])  # 이미지 라벨을 선택한 사진으로 교체
        self.lb_img.setPixmap(pixmap)

        try:
            img = Image.open(self.path[0])
            self.classification(self.preprocessing(img))

        except:
            print('error : {}'.format(self.path[0]))


    def Changed_Num(self, num):

        df_train, df_test = np.load('./datasets/train_test_{}.pkl', allow_pickle=True)
        X_test = df_test.waferMap

        if 0 <= num < len(X_test):
            img_array = X_test[num]

            height, width = img_array.shape
            bytes_per_line = width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)

            self.lb_img.setPixmap(pixmap)
        else:
            self.lb_img.setText("Wrong Index")

        # df_train, df_test = np.load(
        #     './datasets/train_test_{}.pkl', allow_pickle=True)
        # X_test = df_test.waferMap
        # if 0 <= int <= len(X_test):
        #     test_img = X_test[int]  # 이미지 라벨을 선택한 사진으로 교체
        #     self.lb_img.setPixmap(test_img)
        # else:
        #     self.lb_img.setText("Wrong Index")


    def Changed_Str(self, lb):
        try:
            lb = int(lb)
            self.Changed_Num(lb)
        except:
            labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

            if lb in labels:
                self.choice_ramdom_index(labels)
                self.lb_img.setPixmap(self.random_waferMap)
                self.classification(self.random_waferMap)

            else:
                self.lb_img.setText("Wrong Category")

    def classification(self, img):
        pred = self.model.predict(img)
        predicted_prob = np.max(pred)
        pred_str = f'Probability: {predicted_prob * 100:.2f}%'
        self.lb_pred.setText(pred_str)

        labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
        predicted_label = labels[np.argmax(pred)]
        self.lb_label.setText(f'Predicted Label: {predicted_label}')

    def choice_ramdom_index(self, label):
        df_train, df_test = np.load('./datasets/train_test_{}.pkl', allow_pickle=True)
        category = df_test[df_test['failureNum'] == label]
        random_waferMap = np.random.choice(category['waferMap'])

        return random_waferMap

    def preprocessing(self, img):
        img = img.convert('RGB')
        data = np.asarray(img)
        data = data / 255
        data = data.resize((37, 37))
        print(data)
        data = data.reshape(-1, 37, 37, 1)
        return data


if __name__ == '__main__':  # 메인
    app = QApplication(sys.argv)  # 객체 생성
    MainWindow = Exam()
    MainWindow.show()  # 창를 화면에 출력
    sys.exit(app.exec_())  # 창 화면 출력을 유지하는 함수
