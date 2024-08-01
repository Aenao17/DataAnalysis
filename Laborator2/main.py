import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import os
import nltk
import random
import rowordnet as rwn
from collections import Counter
import spacy

class problema1:
    def __init__(self):
        self.data = self.citire_date()

    def citire_date(self):
        data = pd.read_csv('data/employees.csv', delimiter=',', header='infer')
        return data

    def ex1a(self):
        print("Start")
        print("1a. Numarul de angajati este: " + str(self.data['First Name'].count()))
        print("    Numarul de angajati fara nume: " + str(self.data['First Name'].isnull().sum()))
        print("")
        print("    Numarul si tipul proprietatilor: ")
        print("            " + str(self.data.shape[1]))
        print(str(self.data.dtypes))
        print("")
        print("    Numarul angajatilor cu date complete: " + str(self.data.shape[0] - self.data[self.data.isnull().any(axis=1)].count().max()))
        print("")
        print("    Valori minime, maxime, medii: ")
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64'])
        for nume in numeric_columns:
            minim = self.data[nume].min()
            maxim = self.data[nume].max()
            medie = self.data[nume].mean()
            print("            " + nume + ": (MINIM) " + str(minim) + ", (MAXIM) " + str(maxim) + ", (MEDIU) " + str(medie))
        print("")
        print("     Numarul de valori posibile pentru proprietatile nenumerice: ")
        nenumeric_columns = self.data.select_dtypes(exclude=['float64', 'int64'])
        for nume in nenumeric_columns:
            print(nume + ": " + str(self.data[nume].nunique()))
        print("")
        print("Afisam liniile cu valori lipsa: ")
        print(self.data[self.data.isnull().any(axis=1)])
        print("Cum rezolvam? Stergem liniile cu valori lipsa sau le completam cu valori medii")
        print(self.data.dropna(axis=0, how='any'))
        print("End")

    def ex1bSalary(self):

        #y-axis = how many employees have a salary in the range represented by bars on the histogram
        var_data = self.data['Salary']
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        ax.hist(var_data, bins = 10, edgecolor='black')

        ax.set_title("Distributia salariilor")
        ax.set_xlabel("Salariu")
        ax.set_ylabel("Numar angajati")
        plt.show()

    def ex1bEchipa(self):
        salary_cat = ['Low', 'Medium', 'High']
        salary_bins = [0, 50000, 100000, float('inf')]

        #grupam pe categorii
        self.data['Salary Category'] = pd.cut(self.data['Salary'], bins=salary_bins, labels=salary_cat, right=False)

        distribution = self.data.groupby(['Team', 'Salary Category'], observed=True).size().unstack(fill_value=0)

        distribution.plot(kind='bar', stacked=True,figsize=(12, 6))

        plt.title('Distributia salariilor pe echipe')
        plt.xlabel('Echipa')
        plt.ylabel('Numar angajati')
        plt.xticks(rotation=45, ha='right')

        plt.legend(title='Salariu', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def ex1bOutlieri(self):
        '''
        Folosim boxplot si Quartils
        Q1 = valoarea pentru care 25% din date sunt mai mici
        Q2 = mediana
        Q3 = valoarea pentru care 75% din date sunt mai mici
        Q4 = cele mai mari 25% din date
        IQR = Q3 - Q1
        Limita inferioara = Q1 - 1.5 * IQR
        Limita superioara = Q3 + 1.5 * IQR
        Outlieri = valori mai mici decat limita inferioara sau mai mari decat limita superioara
        '''
        salary = self.data['Salary']
        min_salary = salary.min()
        max_salary = salary.max()
        mean_salary = salary.mean()
        median_salary = salary.median()
        mod_salary = salary.mode()[0]

        print("Salariu minim: " + str(min_salary) + " Salariu maxim: " + str(max_salary) + " Salariu mediu: " + str(mean_salary) + " Salariu mediana: " + str(median_salary) + " Salariu mod: " + str(mod_salary))

        fig,ax = plt.subplots(figsize=(15,12))
        outlier = 40000
        outlier_data = salary[salary < outlier]
        filier_props = dict(marker='o', markerfacecolor='r', markersize=8, linestyle='none')

        ax.boxplot(salary, vert=False, flierprops=filier_props, showfliers=True, whis=(outlier/max_salary))
        ax.set_xlabel('Salariu')
        ax.set_title('Boxplot salarii Outlieri')
        plt.show()

    def normalizare(self):
        scaler = MinMaxScaler()
        df_normalized = self.data[['Salary','Bonus %', 'Team']].copy()

        df_normalized[['Salary','Bonus %']] = scaler.fit_transform(df_normalized[['Salary','Bonus %']])
        df_normalized['Team'] = df_normalized['Team'] + 'Salary: ' + df_normalized['Salary'].astype(str) + ' Bonus: ' + df_normalized['Bonus %'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))
        df_normalized.plot(x='Team', y=['Salary','Bonus %'], kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.show()

class problema2:
    def __init__(self):
        self.folder = 'data/images'
    def pb2a(self):
        '''
        Vizualizare o imagine
        '''
        imagini_all = os.listdir(self.folder)
        imagini = [f for f in imagini_all if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]

        if len(imagini) == 0:
            print("Nu exista imagini in folder")
            return

        imagine_aleasa = random.choice(imagini)
        image_path = os.path.join(self.folder, imagine_aleasa)

        imagine = plt.imread(image_path)
        plt.imshow(imagine)
        plt.axis('off')
        plt.show()

    def pb2b(self):
        '''
        daca imaginile nu aceeasi dimensiune,
        sa se redimensioneze toate la 128 x 128 pixeli si sa se vizualizeze imaginile intr-un cadru tabelar.
        '''
        images = []
        for name in os.listdir(self.folder):
            if name.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                img_path = os.path.join(self.folder, name)
                img = Image.open(img_path)
                img = img.resize((128, 128))
                images.append(img)
        nr_imagini = len(images)
        nr_coloane = 4
        nr_linii = (nr_imagini + nr_coloane - 1) // nr_coloane
        fig, ax = plt.subplots(nr_linii, nr_coloane, figsize=(20, 20))

        for i in range(nr_linii):
            for j in range(nr_coloane):
                index = i * nr_coloane + j
                if index < nr_imagini:
                    ax[i, j].imshow(images[index])
                    ax[i, j].axis('off')
                else:
                    ax[i, j].axis('off')
        plt.tight_layout()
        plt.show()

    def pb2c(self):
        '''
        Vizualizare imagini in format grayscale
        '''
        images = []
        for name in os.listdir(self.folder):
            if name.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                img_path = os.path.join(self.folder, name)
                img = Image.open(img_path)
                img = img.convert('L')
                images.append(img)

        nr_imagini = len(images)
        nr_randuri = (nr_imagini+ 3) // 4
        fig, ax = plt.subplots(nr_randuri, 4, figsize=(20, 20))
        for i in range(nr_randuri):
            for j in range(4):
                index = i * 4 + j
                if index < nr_imagini:
                    ax[i, j].imshow(images[index], cmap='gray')
                    ax[i, j].axis('off')
                else:
                    ax[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    def pb2d(self):
        '''
        Blureaza o imagine si o afiseaza before and after
        '''
        imagini_all = os.listdir(self.folder)
        imagini = [f for f in imagini_all if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]

        if len(imagini) == 0:
            print("Nu exista imagini in folder")
            return

        imagine_aleasa = random.choice(imagini)
        image_path = os.path.join(self.folder, imagine_aleasa)
        img = Image.open(image_path)
        img_blur = img.filter(ImageFilter.BLUR)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title('Imagine originala')
        ax[1].imshow(img_blur)
        ax[1].axis('off')
        ax[1].set_title('Imagine blurata')
        plt.show()

    def pb2e(self):
        '''
        Sa se identifice muchiile dintr-o imagine si sa se afiseze imaginea initiala si imaginea cu muchii
        '''
        imagini_all = os.listdir(self.folder)
        imagini = [f for f in imagini_all if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]

        if len(imagini) == 0:
            print("Nu exista imagini in folder")
            return

        imagine_aleasa = random.choice(imagini)
        image_path = os.path.join(self.folder, imagine_aleasa)
        img = Image.open(image_path)
        img_edges = img.filter(ImageFilter.FIND_EDGES)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title('Imagine originala')
        ax[1].imshow(img_edges)
        ax[1].axis('off')
        ax[1].set_title('Imagine cu muchii')
        plt.show()

    def normalizare_pixeli(self):
        '''
        Normalizare pixeli before and after
        '''

        normalized_images = []
        for img in os.listdir(self.folder):
            if img.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                img_path = os.path.join(self.folder, img)
                picture = Image.open(img_path)
                img_array = np.asarray(picture, dtype=np.float32)
                img_n = img_array / 255.0
                normalized_images.append(img_n)

                fig, ax = plt.subplots(1, 2, figsize=(10, 10))
                ax[0].imshow(picture)
                ax[0].axis('off')
                ax[0].set_title('Imagine originala')
                ax[1].imshow(img_n)
                ax[1].axis('off')
                ax[1].set_title('Imagine normalizata')
                plt.imshow(img_n)
                plt.axis('off')
                plt.show()

class problema3:

    def __init__(self):
        self.text = "data/texts.txt"

    def ex3a(self):
        '''
        Numarul de propozitii din text
        '''
        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()

        propozitii = nltk.sent_tokenize(text)
        print("Numarul de propozitii din text: " + str(len(propozitii)))


    def ex3b(self):
        '''
        Numarul de cuvinte din text
        '''

        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()
        word = nltk.word_tokenize(text)
        print("Numarul de cuvinte din text: " + str(len(word)))

    def ex3c(self):
        '''
        Numarul de cuvinte unice din text
        '''
        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()
        word = nltk.word_tokenize(text)
        print("Numarul de cuvinte unice din text: " + str(len(set(word))))

    def ex3d(self):
        '''
        Cele mai lungi si cele mai scurte cuvinte din text
        '''

        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()

        word = nltk.word_tokenize(text)
        word = [w for w in word if w.isalpha()]
        word = [w.lower() for w in word]
        words = []
        for w in word:
            if w not in words:
                words.append(w)
        word = words

        lungime_cuvinte = [len(w) for w in word]
        lungime_maxima = max(lungime_cuvinte)
        lungime_minima = min(lungime_cuvinte)
        cuvinte_maxime = [w for w in word if len(w) == lungime_maxima]
        cuvinte_minime = [w for w in word if len(w) == lungime_minima]
        print("Cele mai lungi cuvinte: " + str(cuvinte_maxime))
        print("Cele mai scurte cuvinte: " + str(cuvinte_minime))

    def ex3e(self):
        '''
        Textul fara diactrice
        '''

        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace('ă', 'a')
        text = text.replace('â', 'a')
        text = text.replace('î', 'i')
        text = text.replace('ș', 's')
        text = text.replace('ț', 't')
        text = text.replace('Ă', 'A')
        text = text.replace('Â', 'A')
        text = text.replace('Î', 'I')
        text = text.replace('Ș', 'S')
        text = text.replace('Ț', 'T')
        print(text)

    def ex3f(self):
        '''
        Sinonimele pentru cel mai lung cuvant
        '''
        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()

        word = nltk.word_tokenize(text)
        word = [w for w in word if w.isalpha()]
        word = [w.lower() for w in word]
        words = []
        for w in word:
            if w not in words:
                words.append(w)
        word = words

        lungime_cuvinte = [len(w) for w in word]
        lungime_maxima = max(lungime_cuvinte)
        cuvinte_maxime = [w for w in word if len(w) == lungime_maxima]
        cuvant = cuvinte_maxime[0]
        print("Cuvantul cu cea mai mare lungime: " + cuvant)

        wn = rwn.RoWordNet()
        cuvant = self.lemmatize_word(cuvant)
        lema = nltk.stem.WordNetLemmatizer().lemmatize(cuvant)
        synset_ids = wn.synsets(literal=lema)
        print(synset_ids)

        # synonyms = set()
        for sin in synset_ids:
            synset_object = wn.synsets(sin)
            print(synset_object)
            synset_object = wn(sin)
            print(synset_object)

    def lemmatize_word(self, word):
        nlp = spacy.load('ro_core_news_sm')
        doc = nlp(word)
        lemmatized_word = doc[0].lemma_
        return lemmatized_word

    def normalizare(self):
        with open(self.text, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = nltk.sent_tokenize(text)
        random_sentence = random.choice(sentences)
        words = nltk.word_tokenize(random_sentence)
        words_c = Counter(words)
        total_words = len(words)

        normalized_frequency = {}
        for word, count in words_c.items():
            normalized_frequency[word] = count / total_words

        print(normalized_frequency)

class Menu:
    def __init__(self):
        self.problema1 = problema1()
        self.problema2 = problema2()
        self.problema3 = problema3()

    def show_menu(self):
        while True:
            print("1. Problema 1")
            print("2. Problema 2")
            print("3. Problema 3")
            print("0. Exit")
            option = input("Enter option: ")
            if(option == "1"):
                self.submenu1()
            elif(option == "2"):
                self.submenu2()
            elif(option == "3"):
                self.submenu3()
            elif(option == "0"):
                break
            else:
                print("Invalid option")

    def submenu1(self):
            while True:
                print("1. Ex1a")
                print("2. Ex1bSalary")
                print("3. Ex1bEchipa")
                print("4. Ex1bOutlieri")
                print("5. Normalizare")
                print("0. Exit")
                option = input("Enter option: ")
                if(option == "1"):
                    self.problema1.ex1a()
                elif(option == "2"):
                    self.problema1.ex1bSalary()
                elif(option == "3"):
                    self.problema1.ex1bEchipa()
                elif(option == "4"):
                    self.problema1.ex1bOutlieri()
                elif(option == "5"):
                    self.problema1.normalizare()
                elif(option == "0"):
                    break
                else:
                    print("Invalid option")

    def submenu2(self):
            while True:
                print("1. Ex2a")
                print("2. Ex2b")
                print("3. Ex2c")
                print("4. Ex2d")
                print("5. Ex2e")
                print("6. Normalizare pixeli")
                print("0. Exit")
                option = input("Enter option: ")
                if(option == "1"):
                    self.problema2.pb2a()
                elif(option == "2"):
                    self.problema2.pb2b()
                elif(option == "3"):
                    self.problema2.pb2c()
                elif(option == "4"):
                    self.problema2.pb2d()
                elif(option == "5"):
                    self.problema2.pb2e()
                elif(option == "6"):
                    self.problema2.normalizare_pixeli()
                elif(option == "0"):
                    break
                else:
                    print("Invalid option")

    def submenu3(self):
            while True:
                print("1. Ex3a")
                print("2. Ex3b")
                print("3. Ex3c")
                print("4. Ex3d")
                print("5. Ex3e")
                print("6. Ex3f")
                print("7. Normalizare")
                print("0. Exit")
                option = input("Enter option: ")
                if(option == "1"):
                    self.problema3.ex3a()
                elif(option == "2"):
                    self.problema3.ex3b()
                elif(option == "3"):
                    self.problema3.ex3c()
                elif(option == "4"):
                    self.problema3.ex3d()
                elif(option == "5"):
                    self.problema3.ex3e()
                elif(option == "6"):
                    self.problema3.ex3f()
                elif(option == "7"):
                    self.problema3.normalizare()
                elif(option == "0"):
                    break
                else:
                    print("Invalid option")
spacy.cli.download("ro_core_news_sm")
spacy.load('ro_core_news_sm')
Menu().show_menu()