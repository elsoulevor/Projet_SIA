##Convertie un dataset en fichier csv
import xlsxwriter
import os

PATH=r"C:/STOCK/Etude/Ensea3A/Projet3A"
PATH_DATA=PATH+'/DatasetSMALL'

workbook = xlsxwriter.Workbook(PATH+'/dataSmall.xlsx')
worksheet = workbook.add_worksheet()

actions=os.listdir(PATH_DATA) #liste des actions
idx=2 #index
worksheet.write('A1', "video_name") #data
worksheet.write('B1', "tag")
for action in actions:
    VideoList=os.listdir(PATH_DATA+'/'+action) #liste des vidéos
    for video in VideoList:
        worksheet.write('A'+str(idx), video) #data
        worksheet.write('B'+str(idx), action) #label
        idx+=1
workbook.close()