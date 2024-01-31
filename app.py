from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5 import uic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys, warnings
import pandas as pd

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

form_window = uic.loadUiType('./classification_test.ui')[0]


class Exam(QWidget, form_window):  # 클래스 생성
    def __init__(self):
        super().__init__()  # 부모 클래스
        self.setupUi(self)
        self.setFixedWidth(450)
        self.setFixedHeight(700)

        # Plot
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # 모델 불러오기
        model_path = './models/CNN_Copy_0.932.h5'
        self.model = load_model(model_path)

        # 초기 화면 설정
        self.lb_img.setText("Wafer Defect Classification Program\nEnter Category or Index Num")

        # 함수 연결
        self.btn_search.clicked.connect(self.Btn_search_clicked)
        self.cb.activated[str].connect(self.Choose_category)
        self.le.textChanged.connect(self.Le_changed)

        # 변수 초기화
        self.labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-Full']
        self.label_mapping = {0: 'Normal', 1: 'Center', 2: 'Donut', 3: 'Edge-Loc', 4: 'Edge-Ring', 5: 'Loc',
                              6: 'Random', 7: 'Scratch', 8: 'Near-full'}
        self.X_train = None
        self.X_test = None
        self.x_train = None
        self.x_test = None
        self.Y_train = None
        self.Y_test = None
        self.y_train = None
        self.y_test = None
        self.label_mapping = None

        self.lb_pred.setText('')
        self.lb_label.setText('')

    def Pickle_load(self):
        self.X_train, self.X_test, self.x_train, self.x_test, self.Y_train, self.Y_test, self.y_train, self.y_test = np.load(
            './datasets/train_test_Copy_0.932.pkl', allow_pickle=True)
        self.Y_test = self.Y_test.reset_index(drop=True)


    def Btn_search_clicked(self):
        try:
            if self.le.text().isdigit():
                label = int(self.le.text())
                self.Find_index(label)
                print('debug : It is number')
            else:
                label = str(self.le.text())
                self.Find_category(label)
                print('debug : It is str')
        except Exception as e:
            print('error:', e)
            print('error : btn_search_clicked')

    def Find_index(self, label):
        try:
            self.Pickle_load()
            # 존재하는 인덱스 값인 경우
            if label < len(self.x_test):
                print('test2')
            # 인덱스 값이 없을때
            else: print('Wrong Index Num')

        except Exception as e:
            print('error:', e)
            print('error : Find_index')

    def Find_category(self, label):
        try:
            self.Pickle_load()
            if label.title() in self.labels:
                print('Debug: Find_category')
                # 해당 카테고리 중 랜덤 인덱스 하나 고르기
                numeric_label = self.labels.index(label.title())
                print('numeric_label: ', numeric_label)
                selected_rows = self.Y_test[self.Y_test == numeric_label]
                random_index = np.random.choice(selected_rows.index)
                print('random_index: ', random_index)
                # 이미지 플롯
                self.Plot_wafer_img(random_index)
                # 결함 종류 예측 및 정확도 출력
                self.Predict(random_index)
            elif label == 'Category':
                # 다른 동작 대기
                pass
            else:
                print('debug: Find_category - else')
                self.lb_img.setText("Wrong Category")

        except Exception as e:
            print('error:', e)
            print('error : Find_category')

    def Choose_category(self, label):
        try:
            self.le.clear()
            print(label)
            self.Find_category(label)
        except Exception as e:
            print('error:', e)
            print('error : Choose_combobox')

    def Le_changed(self):
        # 텍스트가 변경되면 ComboBox 초기화
        self.cb.setCurrentIndex(0)

    def Plot_wafer_img(self, index):
        try:
            if len(plt.get_fignums()) > 0:
                plt.close()

            self.lb_img.setText("")
            self.Pickle_load()
            true_label = self.labels[self.Y_test[index]]

            if not self.fig.get_axes():
                ax = self.fig.add_subplot(111)
            else:
                ax = self.fig.get_axes()[0]
                ax.clear()

            ax.imshow(self.X_test[index], cmap=plt.cm.gray, vmin=0, vmax=2)

            ax.set_title(f'True Label: {true_label}')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()

            q_img = QImage(self.canvas.buffer_rgba(), self.canvas.get_width_height()[0], self.canvas.get_width_height()[1],
                           QImage.Format_ARGB32)
            pixmap = QPixmap.fromImage(q_img)

            # QPixmap을 QLabel에 설정하여 이미지 표시
            self.lb_img.setPixmap(pixmap)

            # # 이미지 플로팅 및 레이블 확인
            # plt.imshow(self.X_test[index], cmap='gray')  # cmap = 'gray': 흑백 처리
            # plt.title("label: [{}]".format(true_label))
            # plt.xlabel(self.Y_test[index])
            # plt.show()

        except Exception as e:
            print('error:', e)
            print('error : plot_img')

    def Predict(self, index):
        try:
            self.Pickle_load()
            print(self.X_test[index].shape)
            pred = self.model.predict(self.X_test[index].reshape(1, 38, 38, 1))
            print(pred)
            max_pred = np.max(pred)
            max_percentage = max_pred * 100
            self.lb_pred.setText(f"{max_percentage:.3f}%")

            pred_label = self.labels[np.argmax(pred)]
            print(pred_label)
            self.lb_label.setText(pred_label)
        except Exception as e:
            print('error:', e)
            print('error : Predict')


if __name__ == '__main__':  # 메인
    app = QApplication(sys.argv)  # 객체 생성
    MainWindow = Exam()
    MainWindow.show()  # 창를 화면에 출력
    sys.exit(app.exec_())  # 창 화면 출력을 유지하는 함수
