import librosa
import os
import json
dataset_path="C:\\Users\\Alexandra\\Desktop\\Lucrarea_licenta\\inregistrari"
#dataset_path=os.getcwd()+"\\inregisrari"
#data_path = "C:\\Users\\Alexandra\\Desktop\\Lucrarea_licenta\\Retea_neuronala1\\test"
json_file = "data2.json"


def dataset_prepare(data_path):
    """
    Aceasta  functie  va  extrage informatiile  necesare pentru atrenarea retelei neuronale.
    :param data_path: calea  catre  fisierul unde se afla datele
    :param sampels_to_consider: nr de esantioane considerate
    :return: dictionaries_data
            mappings is  a list used to map different words to  a number - index
            labels  the target output what we expected
            MFCCs expected outputs that  we need
            files a list that contains the  names of the files
    """
    sampels_to_consider = 22050  # sunt necesare 22050 de esantioane pentru o secunda
    # de cate ori pe secunda este esantionat un semnal / frecventa esantioanelor
    n_mfccs=13
    hop_length = 512
    n_fft = 2048  # n_fft   cat de mare  sa fie  the window  pentru   Fast Fourier  Transform ,
    # pe care vrem s-o cosideram
    # mappings is  a list used to map different words to  a number - index
    # labels  the target output what we expected
    # MFCCs expected outputs that  we need
    # files a list that contains the  names of the files
    dictionaries_data = {
        "mappings": [],
        "labels": [],
        "MFCCs_coeficients": [],
        "files": [],
    }

    for i, (root, dirs, files) in enumerate(os.walk(data_path)):
        # i=0 te afli in  fisierul  test
        if i > 0:
            # numele  cuvantului inregistrat
            list_path = root.split("\\")  # lista care contine  calea unde se afla  inregistrarile
            # [.... , no] ,[....,go]- ultimul element este numele  cuvantului inregistrat
            dictionaries_data["mappings"].append(list_path[len(list_path) - 1])
            print("\n Procesarea cuvantului:{}".format(list_path[len(list_path)-1]))
            for curent_file in files:
                dictionaries_data["labels"].append(i - 1)
                file_path = os.path.join(root, curent_file)
                dictionaries_data["files"].append(file_path)
                signal,sample_rate=librosa.load(file_path)
                if len(signal)>= sampels_to_consider:
                    signal= signal[:sampels_to_consider]
                    #verficare MFCCs
                    MFCCs=librosa.feature.mfcc(signal,sample_rate,n_mfcc=n_mfccs,n_fft=n_fft,hop_length=hop_length)
                    dictionaries_data["MFCCs_coeficients"].append(MFCCs.T.tolist())
                    print("Fisierul {} : indexul {}".format(file_path,i-1))
    return dictionaries_data


def write_dataset(json_path,data):
    """
    Aceasta  functie va  scrie  ce este stocat in variabila data in  fisierul
    aflat la calea  json_path.

    :param json_path:
    :param data:
    :return: Fara valoare returnată

    """
    with open(json_path,"w") as fl:
        json.dump(data,fl,indent=2)

if __name__=="__main__":
    data=dataset_prepare(dataset_path)
    write_dataset(json_file,data)


