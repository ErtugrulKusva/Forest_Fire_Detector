#AUTHOR:ERTUĞRUL KUŞVA/ EMAIL: kusvaertugrul@gmail.com
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap
import sys
import cv2 #opencv-python==4.5.3 
import time
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import * 
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                                QVBoxLayout, QWidget)
from PyQt5.QtCore import QProcess
import numpy as np
import sqlite3
from numpy.core.records import array



def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

stop_threads = False
class VideoThread(QThread):
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    frame_counter=0
    class_names = []
    with open("backup/coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet("backup/yolov4.ertugrul", "backup/Ertugrul.cfg")
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU")
    except print(0):
        print("Using CPU")

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416,416), scale=1/255, swapRB=True)
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    ekran_goruntusu_ates_signal = pyqtSignal(np.ndarray)
    ekran_goruntusu_duman_signal = pyqtSignal(np.ndarray)
    konsol_signal=pyqtSignal(str)
    nesne_ismi_signal = pyqtSignal(str,int)
    def run(self):
        global stop_threads
        vc = cv2.VideoCapture("yangin1080.mp4")
        
       
        
        while cv2.waitKey(1) < 1:
            (grabbed, frame) = vc.read()
            if grabbed:
                frame=self.pencere_icine_al(frame)
                self.change_pixmap_signal.emit(frame)
                if stop_threads: 
                    break
        vc.release()

    def pencere_icine_al(self,frame):
        
        start = time.time()
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        end = time.time()

        

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            clName=self.class_names[classid[0]]
            label = "%s : %f" % (clName, score)
            if clName=="ates":
                cv2.rectangle(frame, box, (0,0,255), 3)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if self.frame_counter <200:
                    #cv2.imwrite("ates%d.jpg" % self.counter,frame)
                    self.frame_counter=self.frame_counter +1
                elif self.frame_counter >= 200:
                    cv2.imwrite("ates%d.jpg" % self.frame_counter,frame)
                    self.ekran_goruntusu_ates_signal.emit(frame)
                    self.konsol_signal.emit("Dikkat ateş tespit edildi.")
                    self.frame_counter=0

            if  clName=="duman":
                cv2.rectangle(frame, box, color, 1)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if self.frame_counter <200:
                  
                  self.frame_counter=self.frame_counter +1
                elif self.frame_counter >= 200:
                    cv2.imwrite("duman%d.jpg" % self.frame_counter,frame)
                    self.ekran_goruntusu_duman_signal.emit(frame)
                    self.konsol_signal.emit("Dikkat duman tespit edildi.")
                    self.frame_counter=0
        end_drawing = time.time()
        
        fps_label = "FPS: %.2f (tespit suresi %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return frame

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        
        super(Ui, self).__init__()
        uic.loadUi('FFD1.ui', self)
        self.p = None 
        
        self.display_width = 640
        self.display_height = 480
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.ekran_goruntusu_ates_signal.connect(self.ekran_goruntusu_ates)
        self.thread.ekran_goruntusu_duman_signal.connect(self.ekran_goruntusu_duman)
        self.thread.konsol_signal.connect(self.konsola_yaz)

        self.algilama_butonu.clicked.connect(self.algilama_baslat)
        self.yetkili_kaydet_butonu.clicked.connect(self.yetkili_kaydet)
        self.yetkili_oku()
        self.yetkili_table_widget.itemChanged.connect(self.yetkili_duzenle)


    def algilama_baslat(self):        
        self.thread.start()
    
    def ekran_goruntusu_ates(self,cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        fotograf= QPixmap(qt_img)       
        self.resim_label2.setPixmap(fotograf)
    def ekran_goruntusu_duman(self,cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        fotograf= QPixmap(qt_img)       
        self.resim_label3.setPixmap(fotograf)
    def konsola_yaz(self,konsol_metni):
        t=time.localtime()
        current_time=time.strftime("%H:%M:%S",t)
      
        self.konsol.appendPlainText("["+current_time+"] "+ konsol_metni)
    def yetkili_kaydet(self):

        conn=sqlite3.connect('ffd1.db')
                
        ad_soyad_var=str(self.yetkili_adi_text.text())
        ceptel_no_var=str(self.yetkili_cep_telefonu_text.text())
        mail_adres_var=str(self.yetkili_mail_adresi_text.text())        
        self.konsol_ayarlar.appendPlainText(str(self.yetkili_adi_text.text()))
        self.konsol_ayarlar.appendPlainText(str(self.yetkili_cep_telefonu_text.text()))
        self.konsol_ayarlar.appendPlainText(str(self.yetkili_mail_adresi_text.text()))
        conn.execute("INSERT INTO yetkili_listesi (ad_soyad, ceptel_no, mail_adres) VALUES(?,?,?);",(ad_soyad_var,ceptel_no_var,mail_adres_var))
        conn.commit()
        conn.close()
        self.yetkili_oku()
        self.konsol_ayarlar.appendPlainText("Veri tabanı başarıyla güncellenmiştir.")
    def yetkili_oku(self):
        conn=sqlite3.connect('ffd1.db')
        conn.row_factory = dict_factory
        cur=conn.cursor()
        cur.execute("SELECT id, ad_soyad, ceptel_no, mail_adres from yetkili_listesi")
        tablerow=0
        rows=cur.fetchall()
        
        self.yetkili_table_widget.setRowCount(len(rows))
        for row in rows:

            self.yetkili_table_widget.setItem(tablerow, 0, QtWidgets.QTableWidgetItem(str(row["id"])))
            self.yetkili_table_widget.setItem(tablerow, 1, QTableWidgetItem(row["ad_soyad"]))
            self.yetkili_table_widget.setItem(tablerow, 2, QTableWidgetItem(str(row["ceptel_no"])))
            self.yetkili_table_widget.setItem(tablerow, 3, QTableWidgetItem(row["mail_adres"]))

            tablerow += 1
    def yetkili_duzenle(self,item):
        print("yetkili okuma başarılı:")
        conn=sqlite3.connect('ffd1.db')
        cur=conn.cursor()
        arrayyetkili=["id","ad_soyad","ceptel_no","mail_adres"]
        sutun_no= item.column()
        satir_no=item.row()
        koordinatlar=f"{satir_no},{sutun_no}"
        veritabani_yeri=arrayyetkili[sutun_no]
        item_id=self.yetkili_table_widget.item(satir_no,0).text()
        print(item_id+"yeni eleman:"+item.text())
        sql=f"UPDATE yetkili_listesi set {veritabani_yeri} = '{item.text()}' where id ={item_id}"
        
        cur.execute(sql)
        conn.commit()
        conn.close()

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""                
        qt_img = self.convert_cv_qt(frame)
        self.resim_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    def start_process(self):
        if self.p is None:  # No process running.

            print("İşlem yürütülüyor")
           
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            self.p.start("python detectImgsTiny.py")
    def process_finished(self):
       
        print("Yürütme işlemi tamamlandı.")
        
        self.p = None
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = Ui()
    a.show()
    sys.exit(app.exec_())
