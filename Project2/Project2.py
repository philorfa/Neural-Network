#!/usr/bin/env python
# coding: utf-8

# # Εργαστηριακή Άσκηση 2. Μη επιβλεπόμενη μάθηση. 
# 
# ## Σύστημα συστάσεων βασισμένο στο περιεχόμενο
# ## Σημασιολογική απεικόνιση δεδομένων με χρήση SOM 
# 
# # ΟΜΑΔΑ Α5
# 
# ## Μπακούρος Αριστείδης 03113138
# ## Ορφανουδάκης Φίλιππος 03113140

# In[1]:


get_ipython().system(u'pip install --upgrade pip')
get_ipython().system(u'pip install --upgrade numpy')
get_ipython().system(u'pip install --upgrade pandas')
get_ipython().system(u'pip install --upgrade nltk')
get_ipython().system(u'pip install --upgrade scikit-learn')
get_ipython().system(u'pip install --upgrade somoclu')
import somoclu
import matplotlib
# we will plot inside the notebook and not in separate window
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # απαραίτητα download για τους stemmer/lemmatizer
nltk.download('rslp')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy as sp
import random 
from sklearn.externals import joblib
from sklearn.cluster import KMeans


# ## Εισαγωγή του Dataset

# Το σύνολο δεδομένων με το οποίο θα δουλέψουμε είναι βασισμένο στο [Carnegie Mellon Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/). Πρόκειται για ένα dataset με περίπου 40.000 περιγραφές ταινιών. Η περιγραφή κάθε ταινίας αποτελείται από τον τίτλο της, μια ή περισσότερες ετικέτες που χαρακτηρίζουν το είδος της ταινίας και τέλος τη σύνοψη της υπόθεσής της. Αρχικά εισάγουμε το dataset στο dataframe `df_data_1`: 

# In[2]:


dataset_url = "https://drive.google.com/uc?export=download&id=1PdkVDENX12tQliCk_HtUnAUbfxXvnWuG"
# make direct link for drive docs this way https://www.labnol.org/internet/direct-links-for-google-drive/28356/
df_data_1 = pd.read_csv(dataset_url, sep='\t',  header=None, quoting=3, error_bad_lines=False)


# Σύμφωνα με τον αριθμό της ομάδας μας [εδώ](https://docs.google.com/spreadsheets/d/12AmxMqvjrc0ruNmZYTBNxvnEktbec1DRG64LW7SX4HA/edit?usp=sharing) , μας αντιστοιχει το seed νουμερο 5 
# 
# 
# 1. Το data frame `df_data_2` έχει 128 γραμμές (ομάδες) και 5.000 στήλες. Όπως και σε κάθε ομάδα έτσι και σε εμάς αντιστοιχεί η γραμμή του πίνακα με το `team_seed_number` μας . Η γραμμή αυτή θα περιλαμβάνει 5.000 διαφορετικούς αριθμούς που αντιστοιχούν σε ταινίες του αρχικού dataset. 
# 
# 2. Τρέχουμε τον κώδικα,από όπου θα προκύψουν τα μοναδικά για κάθε ομάδα  titles, categories, catbins, summaries και corpus με τα οποία θα δουλέψουμε.

# In[3]:



# βάλτε το seed που αντιστοιχεί στην ομάδα σας
team_seed_number = 5

movie_seeds_url = "https://drive.google.com/uc?export=download&id=1NkzL6rqv4DYxGY-XTKkmPqEoJ8fNbMk_"
df_data_2 = pd.read_csv(movie_seeds_url, header=None, error_bad_lines=False,encoding='utf-8')

# επιλέγεται 
my_index = df_data_2.iloc[team_seed_number,:].values

titles = df_data_1.iloc[:, [2]].values[my_index] # movie titles (string)
categories = df_data_1.iloc[:, [3]].values[my_index] # movie categories (string)
bins = df_data_1.iloc[:, [4]]
catbins = bins[4].str.split(',', expand=True).values.astype(np.float)[my_index] # movie categories in binary form (1 feature per category)
summaries =  df_data_1.iloc[:, [5]].values[my_index] # movie summaries (string)
corpus = summaries[:,0].tolist() # list form of summaries


# Απαιτούμε η αναγνωση μας να γίνει σε κωδικοποιήση unicode 'utf-8' έτσι ώστε να αποφύγουμε τους μη εκτυπώσιμους χαρακτήρες!

# - Ο πίνακας **titles** περιέχει τους τίτλους των ταινιών. Παράδειγμα: 'Sid and Nancy'.
# - O πίνακας **categories** περιέχει τις κατηγορίες (είδη) της ταινίας υπό τη μορφή string. Παράδειγμα: '"Tragedy",  "Indie",  "Punk rock",  "Addiction Drama",  "Cult",  "Musical",  "Drama",  "Biopic \[feature\]",  "Romantic drama",  "Romance Film",  "Biographical film"'. Παρατηρούμε ότι είναι μια comma separated λίστα strings, με κάθε string να είναι μια κατηγορία.
# - Ο πίνακας **catbins** περιλαμβάνει πάλι τις κατηγορίες των ταινιών αλλά σε δυαδική μορφή ([one hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)). Έχει διαστάσεις 5.000 x 322 (όσες οι διαφορετικές κατηγορίες). Αν η ταινία ανήκει στο συγκεκριμένο είδος η αντίστοιχη στήλη παίρνει την τιμή 1, αλλιώς παίρνει την τιμή 0.
# - Ο πίνακας **summaries** και η λίστα **corpus** περιλαμβάνουν τις συνόψεις των ταινιών (η corpus είναι απλά ο summaries σε μορφή λίστας). Κάθε σύνοψη είναι ένα (συνήθως μεγάλο) string. Παράδειγμα: *'The film is based on the real story of a Soviet Internal Troops soldier who killed his entire unit  as a result of Dedovschina. The plot unfolds mostly on board of the prisoner transport rail car guarded by a unit of paramilitary conscripts.'*
# - Θεωρούμε ως **ID** της κάθε ταινίας τον αριθμό γραμμής της ή το αντίστοιχο στοιχείο της λίστας. Παράδειγμα: για να τυπώσουμε τη σύνοψη της ταινίας με `ID=99` (την εκατοστή) θα γράψουμε `print(corpus[99])`.

# In[4]:


def show_char(ID):
    print(titles[ID])
    print(categories[ID])
    print(catbins[ID])
    print(corpus[ID])
    return


# In[5]:


show_char(5)


# In[6]:


print(corpus[0])


# # Εφαρμογή 1. Υλοποίηση συστήματος συστάσεων ταινιών βασισμένο στο περιεχόμενο
# <img src="http://clture.org/wp-content/uploads/2015/12/Netflix-Streaming-End-of-Year-Posts.jpg" width="50%">

# Η πρώτη εφαρμογή που θα αναπτύξουμε θα είναι ένα [σύστημα συστάσεων](https://en.wikipedia.org/wiki/Recommender_system) ταινιών βασισμένο στο περιεχόμενο (content based recommender system). Τα συστήματα συστάσεων στοχεύουν στο να προτείνουν αυτόματα στο χρήστη αντικείμενα από μια συλλογή τα οποία ιδανικά θέλουμε να βρει ενδιαφέροντα ο χρήστης. Η κατηγοριοποίηση των συστημάτων συστάσεων βασίζεται στο πώς γίνεται η επιλογή (filtering) των συστηνόμενων αντικειμένων. Οι δύο κύριες κατηγορίες είναι η συνεργατική διήθηση (collaborative filtering) όπου το σύστημα προτείνει στο χρήστη αντικείμενα που έχουν αξιολογηθεί θετικά από χρήστες που έχουν παρόμοιο με αυτόν ιστορικό αξιολογήσεων και η διήθηση με βάση το περιεχόμενο (content based filtering), όπου προτείνονται στο χρήστη αντικείμενα με παρόμοιο περιεχόμενο (με βάση κάποια χαρακτηριστικά) με αυτά που έχει προηγουμένως αξιολογήσει θετικά.
# 
# Το σύστημα συστάσεων που θα αναπτύξουμε θα βασίζεται στο **περιεχόμενο** και συγκεκριμένα στις συνόψεις των ταινιών (corpus). 
# 

# In[7]:


len(corpus)


# ### Αρχικά θα πρέπει να επεξεργαστούμε τα κείμενα μας και συγκεκριμενα τα corpus πριν τα εισάγουμε στο σύστημα συστάσεων

# Πρώτο βήμα είναι να μετατρέψουμε τα κεφαλαία γράμματα σε πεζά και να χωρίσουμε τις λέξεις μια προς μια απο το κείμενο , ώστε να φτιάξουμε μια λίστα τα στοιχεία της οποίας θα είναι οι λέξεις.  

# In[8]:


words=[]
corpus_pros = 5000*['']
for i in range(0,len(corpus),1):
    corpus_pros[i] = corpus[i].lower()
    #words.append(nltk.word_tokenize(corpus_pros[i]))
    words.append(nltk.word_tokenize(corpus[i]))


# In[9]:


print(corpus_pros[0])


# In[10]:


print(corpus[0])


# ## Παρατηρήσαμε ότι πειράζουν πολύ το σύστημα μας τα ονόματα , οπότε μια επεξεργασία που εκτελούμε είναι να κάνουμε τα κεφαλαία σε πεζά μονο αν προηγείται τελεία.

# In[11]:


for i in range(0,len(words),1):
    words[i][0]=words[i][0].lower()
    for j in range(1,len(words[i]),1):
        if (words[i][j-1] =='.'):
            words[i][j] = words[i][j].lower()
    


# In[12]:


print(words[0])


# Επόμενο βήμα έιναι να διαγράψουμε τα σημεία στίξης καθώς όπως φαίνεται απο πάνω , εχουν παραμείνει μετα τον διαχωρισμό tokenize 
# Μετά απο αυτό θα σβήσουμε κάποιες συνηθισμένες αγγλικές λέξεις που λόγω της χρήσης τους δεν μπορούν να δώσουν έξτρα πληροφορία στο σύστημα προτάσεων μας αντιθέτως μπορούν να το παραπλανήσουν, επίσης με αυτό τον τρόπο μικραίνουμε τις λίστες μας όπου είναι χρήσιμο στο χρόνο εκτέλεσης ( ίσως είναι λίγο χρονοβόρο αν ξαναεκτελεστεί)

# In[13]:


filtered_words = []
for i in range(0,len(words),1):
    filtered_words.append([word for word in words[i] if word not in stopwords.words('english') + list(string.punctuation)])
    print(i)


# Πρέπει να κάνουμε καλύτερη δουλειά στην αφαίρεση των σημείων στίξης γιατί δεν αφαιρούνται οι λέξεις που περιέχουν περισσότερα από ένα τέτοια σημεία.

# In[14]:


def thorough_filter_upgraded(words):
    filtered_words = []
    for word in words:
        pun = []
        for letter in word:
            pun.append(any(ext in letter for ext in string.punctuation))
        filt=[]
        for i in range(0,len(word),1):
            if not pun[i]:
                filt.append(word[i])
        filtered_words.append(filt)
    return filtered_words

new_filtered_words = thorough_filter_upgraded(filtered_words)


# Έπειτα καθώς τρέχουμε το σύστημα μας , ορίζουμε και εμεις κάποιες λέξεις που χρησιμοποιούνται συχνά , παραπλανούν και τις βγάζουμε από τις λίστες μας
# 

# In[15]:


new_words=[]
for i in range(0,len(new_filtered_words),1):
    stopwordd=['s','plot','film','movie']
    new_words.append([word for word in new_filtered_words[i] if word not in stopwordd])
    


# ### Stemming &amp; Lemmatization
# 
# Για γραμματικούς λόγους, τα κείμενα χρησιμοποιούν διαφορετικές μορφές μιας λέξης, όπως π.χ. *play*, *plays*, *playing*, *played*. Αυτό έχει σαν αποτέλεσμα πως, ενώ αναφερόμαστε σε κάποιο παρόμοιο σημασιολογικό περιεχόμενο, ο υπολογιστής τις καταλαβαίνει ως διαφορετικές και προσθέτει διαστάσεις στην αναπαράσταση. Για να λύσουμε αυτό το πρόβλημα, μπορούμε να χρησιμοποιήσουμε δύο γλωσσολογικούς μετασχηματισμούς, είτε την αφαίρεση της κατάληξης (stemming), είτε τη λημματοποίηση (lemmatization). Ο στόχος, τόσο της αφαίρεσης κατάληξης όσο και της λημματοποίησης, είναι να φέρουν τις διάφορες μορφές της λέξης σε μια κοινή μορφή βάσης. Πιο συγκεκριμένα:
# 
# Η **αφαίρεση της κατάληξης** αναφέρεται σε μια ακατέργαστη ευριστική διαδικασία που απομακρύνει τα άκρα των λέξεων με την ελπίδα να επιτύχει αυτό το στόχο σωστά τις περισσότερες φορές.
# 
# Η **λημματοποίηση** αναφέρεται στην απομάκρυνση της κλίσης των λέξεων και στην επιστροφή της μορφής της λέξης όπως θα τη βρίσκαμε στο λεξικό, με τη χρήση λεξιλογίου και μορφολογικής ανάλυσης των λέξεων. Η μορφή αυτή είναι γνωστή ως λήμμα (*lemma*).

# In[16]:


lem_words = []
stem_words = []
for curr in range(0,len(new_words),1):
    lem_words.append([wordnet_lemmatizer.lemmatize(word) for word in new_words[curr]])
    stem_words.append([porter_stemmer.stem(word) for word in new_words[curr]])
    print(curr)



# Δημιουργούμε το τελικό μας κείμενο , χρησιμοποιώντας τελικά την επεξεργασία stemming

# In[17]:


corpus_final = []
for i in range (0,len(stem_words),1):
    corpus_final.append(' '.join(stem_words[i]))


# In[18]:


print (corpus_final[0])


# In[19]:


print(corpus[0])


# ## Μετατροπή σε TFIDF
# 
# Αφού επεξεργαστήκαμε τα κείμενα μας και τα φέραμε σε κατάλληλη μορφή , μπορούμε να επεξεργαστούμε την πληροφορία συχνότητας που μας δίνει κάθε λέξη , αφου την μετατρέψουμε στο κατάλληλο διάνυσμα.
# 
# Το πρώτο βήμα θα είναι λοιπόν να μετατρέψετε το corpus σε αναπαράσταση tf-idf:

# In[20]:


print("Dimensions before optimizing TfidfVectorizer parameters")
vectorizer = TfidfVectorizer()
tf_idf=vectorizer.fit_transform(corpus_final)
tf_idf_array =tf_idf.toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf.shape)


# Παρπάνω είδαμε τις διαστάσεις του διανύσματος χωρίς καμία βέλτιστοποίηση , Παρακάτω και έπειτα από δοκιμές και με το σύστημα προτάσεων μας καταλήγουμε σε αυτές τις μεταβλητές έτσι ώστε να μην κόβουν την ποιότητα και να δίνουν ένα γρήγορο αποτέλεσμα.

# In[21]:


print("Dimensions with stop_words='english' and min_df=0.014 max_df=0.3")
vectorizer = TfidfVectorizer(stop_words='english',min_df=0.014,max_df=0.3)
tf_idf = vectorizer.fit_transform(corpus_final)
tf_idf_array =tf_idf.toarray() # επιστρέφει sparse matrix, γι'αυτό το κάνουμε .toarray()
print('TF-IDF array shape:', tf_idf_array.shape)


# Μας προτείνεται να χρησιμοποιήσουμε ως κριτήριο την ομοιότητα συνημιτόνου ή cosine similarity

# In[22]:


similarity = cosine_similarity(tf_idf[0], tf_idf[1])
print(similarity)


# In[23]:


cosd = sp.spatial.distance.cosine(tf_idf_array[0], tf_idf_array[1])
print(cosd)


# Στο εργαστήριο όμως χρησιμοποιύμε την spatial cosine distance το οποίο θα χρησιμοποιήσουμε και εδώ , που όπως φαίνεται παραπάνω αλλά και απο την [βιβλιογραφία](https://en.wikipedia.org/wiki/Cosine_similarity) , είναι το συμπλήρωμα του cosine similarity

# In[24]:


def content_recommender(target_movie, max_recommendations):
    print("*** Target Movie ",str(target_movie),"***")
    print("Title: ",titles[target_movie,0],"")
    print("Summary: ",summaries[target_movie,0],"")
    print("Genres: ",categories[target_movie,0],"\n")
    cosd=4999*[0]
    for i in range(0,4999):
        cosd[i] = sp.spatial.distance.cosine(tf_idf_array[target_movie],tf_idf_array[i]) 
    cosd = np.asarray(cosd)
    indices = cosd.argsort().transpose() 
    indices = indices[1:max_recommendations+1] 
    cnt=1
    print("*** ",max_recommendations," most related movies based on content ***\n")
    for i in indices:
        print("*** Recommended movie No. ",str(cnt),"***")
        print("Movie ID: ",str(i),"")
        print("Title: ",titles[i,0],"")
        print("Summary: ",summaries[i,0],"")
        print("Genres: ",categories[i,0],"")
        cnt=cnt+1
    return


# In[25]:


movies_to_check = random.sample(range(0, 4999), 10)
max_recommendations = 5
for target_movie in movies_to_check:
    content_recommender(target_movie,max_recommendations)
    print('\n\n')


# ## Βελτιστοποίηση
# 
# Τρέξαμε το σύστημα μας και ανάλογα με τα αποτελέσματα μας οδηγηθήκαμε σε μερικές βελτιστοποιήσεις :
# - Στην αλλαγή των παραμέτρων του TFIDF, αυξήσαμε το min_df μέχρι το σημείο που δεν υπήρχε θέμα τη ποιότητα , και καταλήξαμε στο 1,4% των κειμένων μας , είναι ορικαή τιμή αλλά με αυτής της κλίμακας τιμές το som δεν ξέφευγει αρκετά απο θέμα χρόνου και μεγέθους , επίσης θέσαμε το max_df σε 30% έτσι ώστε να διώξουμε τις συχνά εμφανιζόμενες λέξεις που δεν προσφέρουν κάποια πληροφορία.
# - Στην προσθήκη λέξεων στο δικό μας stopwords
# - Με την δοκιμή stem ή lem words 
# 
# - !! Αυτό που παρατηρούμε όμως που προβληματίζει και παραπλανεί το σύστημα μας έχει να κάνει με το τη χρήση ονομάτων , ένα παράδειγμα είναι ότι το επίθετο ενός πρωταγωνιστή μας είναι Rocket και το σύστημα προτάσεων μας να προτείνει ταινία με την NASA , για να το βελτιώσουμε κάπως αυτό , μετατρέπουμε τα κεφαλάια σε πεζά μόνο αν προηγείται τελεια(.) , έτσι τα ονόματα διατηρούν την αξία τους . Επίσης με αυτόν τον τρόπο στη δημιουργία του tf-idf θα κοπούν περισσότερες διαστάσεις , αφου πλέον μικρότερη συχνότητα εμφάνισης θα έχει πχ το rocket που πριν είτε όνομα Rocket ήταν είτε rocket το ουσιαστικό ήταν το ίδιο .
# 
# Παρόλα αυτά το πρόβλημα δεν λύνεται τελείως καθώς όπως αναφέρουμε χαρακτηριστικά στις θεματικές ενότητες παρακάτω στο 5ο σετ ταινιων, ένα από τα κριτήρια είναι το όνομα Micheal

# Όπως βλέπουμε και απο τις περιλήψεις των ταινιών , το σύστημα μας ανταπεξέρχεται αρκετά στις απαιτήσεις μας και προτείνει ταινίες με κοινή θεματική στο περιεχόμενο τους , συκγεκριμένα πάνω βλέπουμε το σύστημα να μας δίνει για 10 τυχαίες ταινίες , 5 προτάσεις , βασισμένο στην επεξεργασία κειμένου που κάναμε και στο cosine similarity που εφαρμόσαμε σαν κριτήριο , οι θεματικές που ενώνουν κάθε ταινία ειναι οι εξής :
# 
# 1. Movies with teens or girls as main characters
# 2. Drama
# 3. Crime and ganks,mafia
# 4. Fiction and fantasy
# 5. Movies containing character named Michael and mainly drama
# 6. Murder, film noir
# 7. Biographical films/biopic and life stories
# 8. Drama and family relationships
# 9. Horror 
# 10. Comedies, black comedies

# In[26]:


joblib.dump(tf_idf, 'final_project_2_tf_idf.pkl')


# In[27]:


tf_idf = joblib.load('final_project_2_tf_idf.pkl')


# In[28]:


ls


# # Εφαρμογή 2.  Σημασιολογική απεικόνιση της συλλογής ταινιών με χρήση SOM
# 

# ## Δημιουργία dataset
# Στη δεύτερη εφαρμογή θα βασιστούμε στις τοπολογικές ιδιότητες των Self Organizing Maps (SOM) για να φτιάξουμε ενά χάρτη (grid) δύο διαστάσεων όπου θα απεικονίζονται όλες οι ταινίες της συλλογής της ομάδας με τρόπο χωρικά συνεκτικό ως προς το περιεχόμενο και κυρίως το είδος τους. 
# 
# Η `build_final_set` αρχικά μετατρέπει την αραιή αναπαράσταση tf-idf της εξόδου της `TfidfVectorizer()` σε πυκνή (η [αραιή αναπαράσταση](https://en.wikipedia.org/wiki/Sparse_matrix) έχει τιμές μόνο για τα μη μηδενικά στοιχεία). 
# 
# Στη συνέχεια ενώνει την πυκνή `dense_tf_idf` αναπαράσταση και τις binarized κατηγορίες `catbins` των ταινιών ως επιπλέον στήλες (χαρακτηριστικά). Συνεπώς, κάθε ταινία αναπαρίσταται στο Vector Space Model από τα χαρακτηριστικά του TFIDF και τις κατηγορίες της.
# 
# Τέλος, δέχεται ένα ορισμα για το πόσες ταινίες να επιστρέψει. Αυτό είναι χρήσιμο για να φτιάχνουμε μικρότερα σύνολα δεδομένων ώστε να εκπαιδεύεται ταχύτερα το SOM.
# 

# In[29]:


def build_final_set(doc_limit, tf_idf_only=False):
    dense_tf_idf = tf_idf.toarray()[0:doc_limit,:]
    final_set = np.hstack((dense_tf_idf, catbins[0:doc_limit,:]))
    return np.array(final_set, dtype=np.float32)


# ## Εκπαίδευση χάρτη SOM

# Θα δoυλέψουμε με χάρτη τύπου planar, παραλληλόγραμμου σχήματος νευρώνων με τυχαία αρχικοποίηση (όλα αυτά είναι default). Δοκιμάζουμε διάφορα μεγέθη χάρτη ωστόσο όσο ο αριθμός των νευρώνων μεγαλώνει, μεγαλώνει και ο χρόνος εκπαίδευσης. Για το training δεν θα ξεπεράσουμε τα 100 epochs.
# Σε γενικές γραμμές μπορούμε να βασιστούμε στις default παραμέτρους μέχρι να έχουμε τη δυνατότητα να οπτικοποιήσουμε και να αναλύσουμε ποιοτικά τα αποτελέσματα. 

# In[30]:


def som_train(n_rows,n_columns):
    som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
    get_ipython().magic(u'time som.train(data=final_set, epochs=100)')
    return som


# In[31]:


final_set = build_final_set(5000)


# In[32]:


som=som_train(25,25)


# 
# ## Best matching units
# 
# Μετά από κάθε εκπαίδευση αποθηκεύουμε σε μια μεταβλητή τα best matching units (bmus) για κάθε ταινία. Τα bmus μας δείχνουν σε ποιο νευρώνα ανήκει η κάθε ταινία. Προσοχή: η σύμβαση των συντεταγμένων των νευρώνων είναι (στήλη, γραμμή) δηλαδή το ανάποδο από την Python. Με χρήση της [np.unique](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.unique.html) (μια πολύ χρήσιμη συνάρτηση στην άσκηση) αποθηκεύουμε τα μοναδικά best matching units και τους δείκτες τους (indices) προς τις ταινίες. Σημειώστε ότι μπορεί να έχετε λιγότερα μοναδικά bmus από αριθμό νευρώνων γιατί μπορεί σε κάποιους νευρώνες να μην έχουν ανατεθεί ταινίες. Ως αριθμό νευρώνα θα θεωρήσουμε τον αριθμό γραμμής στον πίνακα μοναδικών bmus.
# 

# In[33]:


surface_state = som.get_surface_state()
bmus = som.get_bmus(surface_state)
bmus_unique = np.unique(bmus,axis=0,return_index=True, return_counts=True)


# ## Ομαδοποίηση (clustering)
# 
# Τυπικά, η ομαδοποίηση σε ένα χάρτη SOM προκύπτει από το unified distance matrix (U-matrix): για κάθε κόμβο υπολογίζεται η μέση απόστασή του από τους γειτονικούς κόμβους. Εάν χρησιμοποιηθεί μπλε χρώμα στις περιοχές του χάρτη όπου η τιμή αυτή είναι χαμηλή (μικρή απόσταση) και κόκκινο εκεί που η τιμή είναι υψηλή (μεγάλη απόσταση), τότε μπορούμε να πούμε ότι οι μπλε περιοχές αποτελούν clusters και οι κόκκινες αποτελούν σύνορα μεταξύ clusters.
# 
# To somoclu δίνει την επιπρόσθετη δυνατότητα να κάνουμε ομαδοποίηση των νευρώνων χρησιμοποιώντας οποιονδήποτε αλγόριθμο ομαδοποίησης του scikit-learn. Στην άσκηση θα χρησιμοποιήσουμε τον k-Means. Για τον αρχικό μας χάρτη δοκιμάζουμε ένα k=20 ή 25. Οι δύο προσεγγίσεις ομαδοποίησης είναι διαφορετικές, οπότε περιμένουμε τα αποτελέσματα να είναι κοντά αλλά όχι τα ίδια.

# In[34]:


n_clusters=25
algorithm = KMeans(n_clusters=n_clusters)
som.cluster(algorithm=algorithm) 
clusters = som.clusters 


# ## Αποθήκευση του SOM
# 
# Επειδή η αρχικοποίηση του SOM γίνεται τυχαία και το clustering είναι και αυτό στοχαστική διαδικασία, οι θέσεις και οι ετικέτες των νευρώνων και των clusters θα είναι διαφορετικές κάθε φορά που τρέχετε τον χάρτη, ακόμα και με τις ίδιες παραμέτρους. Για να αποθηκεύσουμε ένα συγκεκριμένο som και clustering χρησιμοποιούμε και πάλι την `joblib`.

# In[35]:


joblib.dump(som, 'final_som.pkl')
joblib.dump(clusters, 'final_clusters.pkl')


# In[36]:


som = joblib.load('final_som.pkl')
clusters=joblib.load('final_clusters.pkl')


# ### !!Μετά την ανάκληση ενός SOM θυμηθείτε να ακολουθήσετε τη διαδικασία για τα bmus. (new_load = 1 )

# In[37]:


new_load=0
if (new_load==1):
    surface_state = som.get_surface_state()
    bmus = som.get_bmus(surface_state)
    bmus_unique = np.unique(bmus,axis=0,return_index=True, return_counts=True)
    


# ## Οπτικοποίηση U-matrix, clustering και μέγεθος clusters
# 
# Για την εκτύπωση του U-matrix χρησιμοποιούμε τη `view_umatrix` με ορίσματα `bestmatches=True` και `figsize=(15, 15)` ή `figsize=(20, 20)`. Τα διαφορετικά χρώματα που εμφανίζονται στους κόμβους αντιπροσωπεύουν τα διαφορετικά clusters που προκύπτουν από τον k-Means.  Δεν θα τυπώσουμε τις ετικέτες (labels) των δειγμάτων, είναι πολύ μεγάλος ο αριθμός τους.
# 
# Για μια δεύτερη πιο ξεκάθαρη οπτικοποίηση του clustering τυπώνουμε απευθείας τη μεταβλητή `clusters`.
# 
# Τέλος, χρησιμοποιώντας πάλι την `np.unique` (με διαφορετικό όρισμα) και την `np.argsort` (υπάρχουν και άλλοι τρόποι υλοποίησης) εκτυπώνουμε τις ετικέτες των clusters (αριθμοί από 0 έως k-1) και τον αριθμό των νευρώνων σε κάθε cluster, με φθίνουσα ή αύξουσα σειρά ως προς τον αριθμό των νευρώνων. Ουσιαστικά είναι ένα εργαλείο για να βρίσκουμε εύκολα τα μεγάλα και μικρά clusters. 
# 
# Ακολουθεί ένα μη βελτιστοποιημένο παράδειγμα για τις τρεις προηγούμενες εξόδους:
# 
# <img src="https://image.ibb.co/i0tsfR/umatrix_s.jpg" width="35%">
# <img src="https://image.ibb.co/nLgHEm/clusters.png" width="35%">
# 

# In[38]:


som.view_umatrix(bestmatches=True,figsize=(20, 20),colorbar=True)


# In[39]:



clusters_sorted = np.unique(clusters, return_index = True, return_counts = True)
print ('Clusters sorted by increasing number of neurons:\n',clusters_sorted[0])
print ('Cluster Starting Index:\n',clusters_sorted[1])
print ('Number Of Neurons Per Cluster:\n',clusters_sorted[2])


# ##  Πίνακας Clusters

# In[40]:


for i in range(0,len(clusters),1):
    print(clusters[i])


#                         [ 2  2  2 17 17 14 14 14 14 14 14 21 21 21 21 21 16 16 16 16 16 16 16 16  16]
#                         [ 2  2  2 17 17 14 14 14 14 14 14 14 21 21 21 21 21 16 16 16 16 16 16 16  16]
#                         [ 2  2 12 12 12 14 14 14 14 14 14 14 21 21 21 21 16 16 16 16 16 16 16 16  3]
#                         [12 12 12 12 12 14 14 14 14 14 14 14 14 14 16 16 16 16 16 16 16 16 16  3  3]
#                         [12 12 12 12 12 12 12 16 14 14 14 14 14 11 11 16 16 16 16 16 16  3  3  3  3]
#                         [12 12 12 12 12 12 12 16 16 14  5 14 14 11 11 11 11 11 11 16 16  3  3  3  3]
#                         [12 12 12 12  9 16  8 16 16 16  6 19 11 11 11 11 11 11 11 11 16 10  3  3  3]
#                         [12 12 12 12  8  8  8  8 16  6  6  6 19 19 11 11 11 11 11 11 11 10 10 10  10]
#                         [12 12 12 15  8  8  8  8  8  6  6 19 19 19 11 11 11 11 11 11 11 10 10 10  10]
#                         [15 15 15 15  8  8  8  8  8  6  6  6 19 19 11 11 11 11 11 11 11 11 10 10  10]
#                         [15 15 15 15 15  8  8  8  6  6  6  6 19 19 19 11 11 11 11 11 11 11 10 10  10]
#                         [15 15 15 15 15  8  8  8  6  6  6  5  5 19  1  1  1  1 11 11 11 10 10 10  10]
#                         [15 15 15 15 15  0  0  0  6  6  5  5  5  1  1  1  1  1 11 11 11 23 10 10  10]
#                         [15 15 15 15  0  0  0  0  0  5  5  5  5  1  1  1  1  1 23 23 23 23 23 10  10]
#                         [15 15 15 18  0  0  0  0  0  5  5  5  5  5  1  1 13  1 23 23 23 23 23 10  10]
#                         [18 18 18 18 18  0  0  0  0  0  5  5 22 22 13 13 13 13 23 23 23 23 23 23  23]
#                         [18 18 18 18 18 18  0  0  0  0  0  0 22 22 22 13 13 13 23 23 23 23 23 23  23]
#                         [18 18 18 18 18 18 18  0  0  0  0  0 22 22 22 13 13 13 23 23 23 23  7 23  23]
#                         [ 4  4  4 18 18 18 18 24 24 24 24 24 22 22 22 13 13 13  9  9  9  7  7  7  7]
#                         [ 4  4  4 18 17 17 24 24 24 24 24 24 22 22 22 13 13  9  9  9  9  7  7  7  7]
#                         [ 4  4  4 17 17 17 17 24 24 24 24 24 22 22 22 22 13  9  9  9  9  7  7  9  7]
#                         [20 20 17 17 17 17 17 17 24 24 24 24 22 22  2  2  9  9  9  9  9  9  9  9  9]
#                         [20 20 20 17 17 17 17 17  9 24 24  2  2  2  2  2  2  9  9  9  9  9  9  9  9]
#                         [20 20 20 20 17 17 17  9  9  9  2  2  2  2  2  2  2  2  9  9  9  9  9  9  9]
#                         [20 20 17 20 17 17 17  9  9  9  2  2  2  2  2  2  2  2  9  9  9  9  9  9  9]

# 
# ## Σημασιολογική ερμηνεία των clusters
# 
# Προκειμένου να μελετήσουμε τις τοπολογικές ιδιότητες του SOM και το αν έχουν ενσωματώσει σημασιολογική πληροφορία για τις ταινίες διαμέσου της διανυσματικής αναπαράστασης με το tf-idf και των κατηγοριών, χρειαζόμαστε ένα κριτήριο ποιοτικής επισκόπησης των clusters. Θα υλοποιήσουμε το εξής κριτήριο: Λαμβάνουμε όρισμα έναν αριθμό (ετικέτα) cluster. Για το cluster αυτό βρίσκουμε όλους τους νευρώνες που του έχουν ανατεθεί από τον k-Means. Για όλους τους νευρώνες αυτούς βρίσκουμε όλες τις ταινίες που τους έχουν ανατεθεί (για τις οποίες αποτελούν bmus). Για όλες αυτές τις ταινίες τυπώνουμε ταξινομημένη τη συνολική στατιστική όλων των ειδών (κατηγοριών) και τις συχνότητές τους. Αν το cluster διαθέτει καλή συνοχή και εξειδίκευση, θα πρέπει κάποιες κατηγορίες να έχουν σαφώς μεγαλύτερη συχνότητα από τις υπόλοιπες. Θα μπορούμε τότε να αναθέσουμε αυτήν/ές την/τις κατηγορία/ες ως ετικέτες κινηματογραφικού είδους στο cluster.
# 
# Μπορείτε να υλοποιήσετε τη συνάρτηση αυτή όπως θέλετε. Μια πιθανή διαδικασία θα μπορούσε να είναι η ακόλουθη:
# 
# 1. Ορίζουμε συνάρτηση `print_categories_stats` που δέχεται ως είσοδο λίστα με ids ταινιών. Δημιουργούμε μια κενή λίστα συνολικών κατηγοριών. Στη συνέχεια, για κάθε ταινία επεξεργαζόμαστε το string `categories` ως εξής: δημιουργούμε μια λίστα διαχωρίζοντας το string κατάλληλα με την `split` και αφαιρούμε τα whitespaces μεταξύ ετικετών με την `strip`. Προσθέτουμε τη λίστα αυτή στη συνολική λίστα κατηγοριών με την `extend`. Τέλος χρησιμοποιούμε πάλι την `np.unique` για να μετρήσουμε συχνότητα μοναδικών ετικετών κατηγοριών και ταξινομούμε με την `np.argsort`. Τυπώνουμε τις κατηγορίες και τις συχνότητες εμφάνισης ταξινομημένα. Χρήσιμες μπορεί να σας φανούν και οι `np.ravel`, `np.nditer`, `np.array2string` και `zip`.

# In[41]:


def print_categories_stats(movies):
    categ = []
    for movie in movies:
        temp = categories[movie][0]
        temp = temp.split('"')
        temp = filter(lambda a: a != '', temp)
        temp = filter(lambda a: a != ',  ',temp)
        categ.extend(temp)

    categ_unique = np.unique(categ, return_counts = True)
    indices = np.argsort(categ_unique[1])
    ran=indices[len(indices)-1]
    print("Cluster Label = ",categ_unique[0][ran],'\n')
    indices = reversed(indices)
    tot = sum(categ_unique[1])
    print ('{:<30} {:<20} {:<20}'.format('Category', 'Percentage', 'NumberOfMovies'))
    print ('-'*66)
    for i in indices:
        print ('{:<30} {:<20} {:<20}'.format(categ_unique[0][i], str(round(categ_unique[1][i]*100.0/tot,3))+'%',str(categ_unique[1][i])))
    return categ_unique[0][ran]


# 2. Ορίζουμε τη βασική μας συνάρτηση `print_cluster_neurons_movies_report` που δέχεται ως όρισμα τον αριθμό ενός cluster. Με τη χρήση της `np.where` μπορούμε να βρούμε τις συντεταγμένες των bmus που αντιστοιχούν στο cluster και με την `column_stack` να φτιάξουμε έναν πίνακα bmus για το cluster. Προσοχή στη σειρά (στήλη - σειρά) στον πίνακα bmus. Για κάθε bmu αυτού του πίνακα ελέγχουμε αν υπάρχει στον πίνακα μοναδικών bmus που έχουμε υπολογίσει στην αρχή συνολικά και αν ναι προσθέτουμε το αντίστοιχο index του νευρώνα σε μια λίστα. Χρήσιμες μπορεί να είναι και οι `np.rollaxis`, `np.append`, `np.asscalar`. Επίσης πιθανώς να πρέπει να υλοποιήσετε ένα κριτήριο ομοιότητας μεταξύ ενός bmu και ενός μοναδικού bmu από τον αρχικό πίνακα bmus.

# In[42]:


def print_cluster_neurons_movies_report(cluster_num):
    print ('Cluster =',cluster_num,'\n')
    new_indices = np.where(clusters == cluster_num)
    new_indices = np.column_stack((new_indices[1],new_indices[0]))
    bmus_right = []
    for i in new_indices:
        if(i in bmus_unique[0]):
            bmus_right.append(i)
    bmus_right = np.array(bmus_right)
    return bmus_right


# 3. Υλοποιούμε μια βοηθητική συνάρτηση `neuron_movies_report`. Λαμβάνει ένα σύνολο νευρώνων από την `print_cluster_neurons_movies_report` και μέσω της `indices` φτιάχνει μια λίστα με το σύνολο ταινιών που ανήκουν σε αυτούς τους νευρώνες. Στο τέλος καλεί με αυτή τη λίστα την `print_categories_stats` που τυπώνει τις στατιστικές των κατηγοριών.

# In[43]:


def neuron_movies_report(cluster_num):
    curr_bmus = print_cluster_neurons_movies_report(cluster_num)
    movies_to_check = []

    for bmu in curr_bmus:
        for id in range(0,len(final_set),1):
            if(np.array_equal(bmu,bmus[id])):
                movies_to_check.append(id)
    cluster_label.append(print_categories_stats(movies_to_check))


# ### !!Τρέχουμε την παραπάνω συνάρτηση!!

# In[44]:


clusters_to_check = range(0,n_clusters,1)
cluster_label=[]
for curr_cluster in clusters_to_check:
    neuron_movies_report(curr_cluster)
    print ('-'*66,'\n\n')


# ## Συμπεράσματα και Ανάλυση τοπολογικών ιδιοτήτων χάρτη SOM

# 1. Αρχικά βλέπουμε ότι στη μεγάλη πλειοψηφία των cluster θα επικρατούν απο τις υπόλοιπες ταινίες με μεγάλη διαφορά 2-3 ταινίες , ένδειξη ότι το cluster διαθέτει καλή συνοχή και εξειδίκευση , ακόμα και τα clusters που ίσως έχουν 4 είδη βλέπουμε ότι αυτά τα είδη είναι απόλυτα συσχετιζόμενα πχ cluster 4 με είδη τα  Action,Thriller,Action/Adventure,Crime Fiction  ή cluster 8 με είδη τα Family Film,Animation,Adventure,Fantasy.
# 
# 2. Ένα είδος που θα εξετάζουμε αρκετά και θα το αναφέρουμε σαν ένα απο τα παραδείγματα μας είναι το Drama το οποίο έχει και πολύ μεγάλη είσοδο, παρατηρούμε ότι το Drama έχει κυριαρχήσει στη κάτω δεξιά γωνία με τα clusters 2,7,9,17
# 
# 3. Άλλο ένα παράδειγμα ειναι το είδος Comedy που όπως βλέπουμε καταλαμβάνει την πανω δεξιά μερία και λίγο απο το κέντρο με τα clusters που το αντιπροσωπεύουν να είναι 1,3,10,11 
# 
# 4. Παρατηρούμε οτι για να γίνει το πέρασμα απο το comedy στο drama υπάρχουν clusters που σαν περιεχόμενο διαθέτουν πολύ υψηλά και τα δύο είδη αλλά και τοπολογικά είναι ανάμεσα τους πχ το cluster 23.
# 
# 5. Άλλη μια μεγάλη κατηγορία ειναι το Action που φαίνεται να έχει καταλάβει την αριστερή πλευρά με clusters τα 0,4,15,18
# 
# 6. Γενικότερα παρατηρούμε μια ομαλη μετάβαση που έχει λογική και ακολουθεί την ανρθώπινη διαίσθηση πχ Action -&gt; Thriller-&gt; Drama. Το οποίο γίνεται με πολύ εξειδικευμένα clusters στην αρχή και στο τέλος,  και με clusters που έχουν είδη να ανταγωνίζονται για την πρώτη θέση ενδιάμεσα.
# 
# 7. Παρατηρούμε πως το Drama με μεγάλη εισόδο απασχολει διαφορετικούς νευρώνες καθως τα σημειά που καταλαμβάνει ειναι κατω δεξιά , περιοχες στο κεντρο και πανω αριστερά. Ενώ είδη με συγκριτικά μικρότερη είσοδο αλλα παρόλα αυτά υπολογίσιμη καταλαμβάνουν λιγότερυς νευρώνες , πχ Short Film με τα clusters 1,5,6 είναι συγκεντρωμένα στο κέντρο.
# 
# 8. Όπως αναφέρουμε και παραπάνω γίνεται μια μετάβαση απο ένα είδος σε ένα άλλο ξένο , αυτό έχει σαν αποτέλεσμα να έχουμε τα είδη που ειναι κοντινά μεταξύ τους σε περιεχόμενο και τοπολογικά κοντά , τέτοια παραδείγματα είναι τo Romance Film με το Romantic Drama ή το Action με το Action Adventure.
# 
# 9. Δυστυχώς υπάρχουν clusters που αντιπροσωπεύουν είδη που δεν έχουν τόσο σχέση μεταξύ τους . Προφανώς δεν έχει επιτευχθεί μια τέλεια μετάβαση καθώς όπως αναφέρεται στην εκφώνηση η τοποθέτηση σε 2 διαστάσεις που να σέβεται μια απόλυτη τοπολογία δεν είναι εφικτή, αφενός γιατί δεν υπάρχει κάποια απόλυτη εξ ορισμού για τα κινηματογραφικά είδη ακόμα και σε πολλές διαστάσεις, αφετέρου γιατί πραγματοποιούμε μείωση διαστατικότητας. 
# 
# 10. Για να φτάσουμε στον τελικό αυτό χάρτη έχουμε πραγματοποιήσει αρκετά μεγάλη μειώση στηλών, για χάρη του χρόνου εκτέλεσης και του μεγέθους του som . Ύστερα απο δοκιμές στον αριθμό των clusters καταλήξαμε σε αυτό το νούμερο για την ομαδοποίηση μας . Ίσως με έναν μεγαλύτερο διάνυσμα tf-idf να είχαμε ένα καλύτερο αποτέλεσμα και στο σύστημα προτάσεων αλλά και στην τελική ομαδοποίηση ειδών , αλλά υπήρχε πολύ μεγάλο κόστος περίπου 15 φορές τον χρόνο εκτέλεσης και 3πλάσιο μέγεθος , που το καθιστούσε απαγορευτικό.

# In[ ]:




