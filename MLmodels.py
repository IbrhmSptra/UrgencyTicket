import mysql.connector as sql  #Connector
import pandas as pd #Table/Dataframe Management
#untuk remove punct (regular expresion)
import re
#perhitungan
import numpy as np
#word embedding lib
from gensim.models import FastText 
#Untuk Tokenization
import nltk
#Export model
from joblib import load
#for sleep cooldown
import time
from tqdm import tqdm
#sys command controller
import sys


#--------------------------------------------------------------------------
#buat Fungsi Koneksi ke Database
def connection() :
    try :
        cn = sql.connect(
            host = "localhost",
            username = "root",
            password = "",
            database = "db_iconsup"
        )
        return cn
    except sql.Error as err :
        raise


#buat Fungsi Read table reports
def read(conn) :
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE urgency is null")
    data = cursor.fetchall()
    reports = pd.DataFrame(data, columns=['id','id_user','pesan','id_services','urgency','created_at','status','status_date'])
    return reports

#buat Fungsi update dan memberikan value Urgency
def add_urgency(conn,urgency,id) :
    cursor = conn.cursor()
    query = "UPDATE reports SET urgency = %s WHERE id = %s"
    values = (urgency,id)
    cursor.execute(query,values)
    conn.commit()

#------------------------------------------------------------------------------------------------

# Fungsi untuk merubah hasil prediski menjadi category high medium low
def formatPredict(pred) :
    if pred == 3 :
        return "High"
    if pred == 2 :
        return "Medium"
    else:
        return "Low"
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------

# Fungsi untuk remove selain huruf dan angka seperti ()[]?}{}':;><,. dll
def rmPunct(text) :
    text = re.sub(r'[^\w\s]',' ',text) #rm punctuation
    text = re.sub(r' +',' ',text) #rm space more than 1
    return text.strip().lower() #rm leading space and make it al lowercase
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
"""
fungsi untuk mengubah word vector menjadi sentence vector
 dengan return np.array 2D berisikan word yang sudah di tokenize dan di lower lalu ubah ke vector satukan dengan cara di rata rata dan satukan lagi menjadi sentence
"""
def vectorize(text,wvmodels) :
     vecs = [wvmodels[word.lower()] for word in nltk.word_tokenize(text)]
     vec = np.mean(vecs,axis=0)
     return vec
#------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    print("======================================================")
    print(f"{'Program Start'.center(50)}")
    print("======================================================")
    print()
    print()
    time.sleep(1)


#------------------------------------------------------------------------------------------------
    """
    Load KNN Models -> untuk memprediksi klasifikasi teks berdasarkan inputan
    """
    print("======================================================")
    print(f"{'Load KNN Models'.center(50)}")
    print("======================================================")
    
    model = load('Text-Classification-Models.joblib') 

    print("======================================================")
    print(f"{'FINISHED!!'.center(50)}")
    print("======================================================")
    print()
    print()
    time.sleep(3)
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
    """
    Load Word Embeding Models -> untuk mengubah teks to vector
    """
    print("======================================================")
    print(f"{'Load Word Embeding Models..'.center(50)}")
    print("======================================================")
    
    path = r"C:\Users\Ibrahim Saputra\OneDrive\Documents\CodingIseng\PHP-PythonComs\wordembedding.fasttext"
    wv = FastText.load(path).wv

    print("======================================================")
    print(f"{'FINISHED!!'.center(50)}")
    print("======================================================")
    print()
    print()
    time.sleep(3)
#------------------------------------------------------------------------------------------------




    print("======================================================")
    print(f"{'On The Loop.. (Watching Databases)'.center(50)}")
    print("======================================================")
    print()
    print()
    time.sleep(1)
    while True:
        try:
            print("======================================================")
            print(f"{'Scanning New Data'.center(50)}")
            print("======================================================")
            time.sleep(1)
            print(".")
            time.sleep(1)
            print(".")
            time.sleep(1)
            print(".")
            #get data from db_iconsup
            cn = connection() #get connection to db
            df = read(cn)
            if len(df) == 0 :
                print("======================================================")
                print(f"{'Not Found Newest Data'.center(50)}")
                print("Data Found : ",len(df))
                print("======================================================")
                print()
                print()
                print("======================================================")
                print(f"{'Program Cooldown 45s'.center(50)}")
                print(f"{'(CTRL + C) to stop the program'.center(50)}")
                print("======================================================")
                time.sleep(45)  # Ganti angka ini sesuai kebutuhan
                continue
            print("======================================================")
            print(f"{'Found Newest Data'.center(50)}")
            print("Data Found : ",len(df))
            print("======================================================")
            print()
            print()
            #get only kolom pesan
            reports = df[['pesan']].copy()
            #Remove Punct
            reports['pesan'] = reports['pesan'].map(rmPunct)
            #Encode
            textVecs = [vectorize(pesan,wv) for pesan in reports['pesan']]
            textVecs = np.array(textVecs)
            print("======================================================")
            print(f"{'Predicting...'.center(50)}")
            print("======================================================")
            #predict
            predict = model.predict(textVecs)
            predict = [formatPredict(pred) for pred in predict]
            time.sleep(3)
            #fill urgency to dataframe
            df['urgency'] = df['urgency'].fillna(pd.Series(predict))
            
            #insert database
            print()
            print()
            print("======================================================")
            print(f"{'Inserting to Database'.center(50)}")
            print("======================================================")
            pbar = tqdm(total=len(df),desc="Inserting..")
            for index, row in df.iterrows() :
                add_urgency(cn,row['urgency'],row['id'])
                time.sleep(0.5)
                pbar.update(1)
            pbar.close()
                
            
            print()
            print()
            print("======================================================")
            print(f"{'Program Cooldown 45s'.center(50)}")
            print(f"{'(CTRL + C) to stop the program'.center(50)}")
            print("======================================================")
            time.sleep(45)  # Ganti angka ini sesuai kebutuhan
        except KeyboardInterrupt:
            break