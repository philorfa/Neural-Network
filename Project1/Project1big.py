#!/usr/bin/env python
# coding: utf-8

# # Α.Στοιχεία Ομάδας

# #      Ομάδα Α5
# 
#    Ορφανουδάκης Φίλιππος (ΑΜ:03113140)
# 
#    Μπακούρος Αριστείδης (ΑΜ:03113138)

# # Β.Εισαγωγή του Dataset

# Το συγκεκριμένο dataset(ISOLET) προήλθε με τον εξής τρόπο: 150 άνθρωποι (αντικείμενα του δείγματος) κλήθηκαν να πουν κάθε γράμμα της αγγλικής αλφαβήτου από 2 φορές. Έτσι, για κάθε αντικείμενο έχουμε 52 (26 γράμματα επί 2 φορές την αλφαβήτα ο καθένας) rows στο dataset μας και 617 attributes συν 1 για την κλάση. Οι κλάσεις παίρνουν τιμές από 1 ως και το 26 και αντιπροσωπεύουν ένα γράμμα της αλφαβήτου(1 για το A, 2 για το B, 3 για το C κ.ο.κ). Αρχικά ήταν χωρισμένοι σε 5 ομάδες των 30 αντικειμένων, η τελευταία ομάδα ήταν το test set, αλλά το αλλάζουμε στη συνέχεια καθώς η άσκηση ζητά 30% test set. Τα attributes αφορούν στοιχεία του γύρω περιβάλλοντος καθώς και ηχητικά χαραχτηριστικά πριν, μετά και κατά τη διάρκεια της προφοράς κάθε γράμματος. Στόχος είναι να προβλεφθεί με όσο το δυνατόν μεγαλύτερη συνέπεια ποιο πράγμα πρόκειται να ειπωθεί βάσει όλων αυτών των χαραχτηριστικών. Δεν υπάρχουν απουσιάζουσες τιμές, έχουν χαθεί μόνο 3 παραδείγματα πιθανώς για προβλήματα στην ηχογράφηση και όλα τα χαραχτηριστικά είναι διατεταγμένα και είναι πραγματικοί αριθμοί από το -1 έως το 1.
# 

# Ξεκινάμε κάνοντας upgrade τις βιβλιοθήκες που θα χρειαστούμε.

# In[1]:


get_ipython().system(u'pip install --upgrade pip #upgrade pip package installer')
get_ipython().system(u'pip install scikit-learn --upgrade #upgrade scikit-learn package')
get_ipython().system(u'pip install numpy --upgrade #upgrade numpy package')
get_ipython().system(u'pip install pandas --upgrade #upgrade pandas package')


# Για να εισάγουμε το dataset μας κατεβάζουμε τα αρχεία isolet1+2+3+4.data και isolet5.data και τα ανεβάζουμε στο ιδιο directory με αυτό το notebook.

# In[2]:


get_ipython().system(u'ls')


# Ακολουθεί μία πρώτη εμφάνιση των 2 αυτών datasets προτού γίνει το concatenation.Επειδή περιέχεται πληροφορία στην 1η γραμμή διαπιστώνουμε ότι δεν έχουμε headers.

# In[3]:


import pandas as pd

df = pd.read_csv("isolet1.data", header=None,na_values = ["?"])
df


# In[4]:


df = pd.read_csv("isolet5.data", header=None,na_values = ["?"])
df


# Ακολούθως, πάμε να κάνουμε το concatenation των 2 αρχείων σε 1 όπως ζητείται. Ταυτόχρονα, για να γίνει ομαλά αυτό και να υπάρχει μία συνέχεια στην αρίθμηση των γραμμών επιλέγουμε να κάνουμε ignore το Index τους δεδομένου ότι δεν περιέχει και κάποια πληροφορία. Αλλάζουμε και το όνομα των κλάσεων, που είναι η τελευταία στήλη σε Letter καθώς αντιπροσωπεύει γράμμα, και έχει 26 ετικέτες, όπως περιγράφηκε παραπάνω.

# In[5]:


isolet1=pd.read_csv("isolet1.data", header=None,na_values = ["?"])
isolet5=pd.read_csv("isolet5.data", header=None,na_values = ["?"])
merged_isolet = pd.concat(([isolet1,isolet5]),axis=0,ignore_index=True)
df=merged_isolet
df = df.rename(columns={617: 'Letter'})
df


# In[6]:


print("Έχουμε", len(df), "δείγματα.")
print("Το καθένα έχει ", df.shape[1]-1, "χαραχτηριστικά, και 1 για την κλάση.")


# Διαδικαστικά τσεκάρουμε για ενδεχόμενες απουσιάζουσες τιμές αν και γνωρίζουμε ότι δεν υπάρχουν από την περιγραφή του dataset

# In[7]:


null_columns=df.columns[df.isnull().any()]
print(df[df.isnull().any(axis=1)][null_columns])


# Σε ό,τι αφορά την ισορροπία του dataset μας, υπάρχει απόλυτη ισορροπία καθώς όλες οι τιμές της κλάσεις εμφανίζονται ακριβώς τις ίδιες φορές.

# Προχωράμε με τον χωρισμό του dataset σε train και test, με το test set να είναι το 30% του dataset όπως ζητείται. Εισάγουμε και τη numpy για μετατροπή του dataframe σε numpy array. Αλλάζουμε και τις ετικέτες των κλάσεων.

# In[8]:


label_names = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
labels_df = df.iloc[:,[617]]
features_df = df.iloc[:,:617]


# In[9]:


import numpy as np
features_df_t=np.asarray(features_df)
labels_df_t=df['Letter']
labels_df_t =list(map(lambda x : x, labels_df_t)) 
from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = train_test_split(features_df_t,labels_df_t, test_size = 0.3)

train_dummy = train
train_labels_dummy = train_labels 
test_dummy = test
test_labels_dummy = test_labels


# # Γ. Baseline classification

# Είμαστε έτοιμοι να εκπαιδεύσουμε τους ταξινομητές χωρίς βελτιστοποιημένες παραμέτρους (με αρχικές τιμές απλά) ή επεξεργασία σε πρώτη φάση. Σημειώνουμε ότι δεν εκτελούμε τον dummyclassifier με strategy constant καθώς με τις 26 διαφορετικές κλάσεις δεν έχει ιδιαίτερο νόημα. Παρακάτω φαίνονται κατά σειρά όλες οι μέθοδοι (dummy classifiers,kNN,gnb,mlp), τυπώνονται τα confussion matrix, f1-macro average,f1-micro average τους για κάθε estimator και παρουσιάζονται τα αποτελέσματα τους με plots σύγκρισης κάθε estimator σχετικά με τους δείτκες f1 τους.

# In[10]:


from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

def function(train_in,train_labels_in,test_in,test_labels_in):
    credit_accuracy = {}

    dc_uniform = DummyClassifier(strategy="uniform")

# με τη μέθοδο fit "εκπαιδεύουμε" τον ταξινομητή στο σύνολο εκπαίδευσης (τα χαρακτηριστικά και τις ετικέτες τους)
    model = dc_uniform.fit(train_in, train_labels_in)

# με τη μέθοδο predict παράγουμε προβλέψεις για τα δεδομένα ελέγχου (είσοδος τα χαρακτηριστικά μόνο)
    preds = dc_uniform.predict(test_in)

# υπολογίζουμε την ακρίβεια του συγκεκριμένου μοντέλου dummy classifier
    credit_accuracy['uniform (random)'] = dc_uniform.score(test_in, test_labels_in)


#################
    print ('Classification report for Dummy Classifier (uniform)')
    cr_dummy_uni = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_uni)

    scores_weighted = {}
    scores_macro = {}
    scores_micro = {}

    scores_weighted['Dummy-Uniform']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Uniform']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Uniform']=precision_recall_fscore_support(test_labels_in,preds,average='micro')


    print ('Confusion Matrix for Dummy Classifier (uniform)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_uni = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_uni))
    
    dc_most_frequent = DummyClassifier(strategy="most_frequent")
    model = dc_most_frequent.fit(train_in, train_labels_in)
    preds = dc_most_frequent.predict(test_in)
    credit_accuracy['most_frequent'] = dc_most_frequent.score(test_in, test_labels_in)

#################
    print ('Classification report for Dummy Classifier (most frequent)')
    cr_dummy_freq = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_freq)

    scores_weighted['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels_in,preds,average='micro')

    print ('Confusion Matrix for Dummy Classifier (most frequent)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_freq = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_freq))
    
    
    
    dc_stratified = DummyClassifier(strategy="stratified")
    model = dc_stratified.fit(train_in, train_labels_in)
    preds = dc_stratified.predict(test_in)
    credit_accuracy['stratified'] = dc_stratified.score(test_in, test_labels_in)

#################
    print ('Classification report for Dummy Classifier (stratified)')
    cr_dummy_strat = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_strat)

    scores_weighted['Dummy-Strat']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Strat']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Strat']=precision_recall_fscore_support(test_labels_in,preds,average='macro')

    print ('Confusion Matrix for Dummy Classifier (stratified)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_strat = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_strat))
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(train_in, train_labels_in)
    knn_preds = knn.predict(test_in)


#################
    print ('Classification report for kNN')
    cr_knn_no = classification_report(test_labels_in,knn_preds, target_names=label_names)
    print (cr_knn_no)

    scores_weighted['kNN-non-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='weighted')
    scores_macro['kNN-non-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='macro')
    scores_micro['kNN-non-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='micro')

    print ('Confusion Matrix for non-optimized kNN ')
    print (confusion_matrix(test_labels_in, knn_preds))

    acc_knn_no = 100*accuracy_score(test_labels_in,knn_preds)
    print ('\nAccuracy percentage of this classifier is %.3f %%\n' % (acc_knn_no))
    
    
    ###############
    
    
    clf = MLPClassifier()
    clf.fit(train_in, train_labels_in)
    preds = clf.predict(test_in)

    print ('Classification report for MLP on initial data')
    cr_mlp_dummy = classification_report(test_labels_in, preds)
    print (cr_mlp_dummy)

    scores_weighted['MLP_dummy']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['MLP_dummy']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['MLP_dummy']=precision_recall_fscore_support(test_labels_in,preds,average='micro')
    
    print ('Confusion matrix for MLP on initial data')
    print (confusion_matrix(test_labels_in, preds))

    acc_mlp_dummy = 100*accuracy_score(test_labels,preds)
    print ('Accuracy percentage of this classifier is %.3f %%' % (acc_mlp_dummy))
    
    
    ######
    
    gnb = GaussianNB()
    gnb.fit(train_in, train_labels_in)
    preds=gnb.predict(test_in)

#################
    print ('Classification report for non_opt Gaussian Naive Bayes Classifier')
    cr_gnb = classification_report(test_labels_in, preds)
    print (cr_gnb)

    scores_weighted['GNB']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['GNB']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['GNB']=precision_recall_fscore_support(test_labels_in,preds,average='micro')

    print ('Confusion Matrix for Gaussian Naive Bayes Classifier')
    print (confusion_matrix(test_labels_in, preds))

    acc_gnb = 100*accuracy_score(test_labels_in,preds)
    print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_gnb))
    
    
    import matplotlib.pyplot as plt

    f1_scores_macro = [item[2] for item in scores_macro.values()]
    f1_scores_micro = [item[2] for item in scores_micro.values()]
    f1_scores_weighted = [item[2] for item in scores_weighted.values()]


    y_pos = np.arange(len(f1_scores_macro))
    plt.barh(y_pos, f1_scores_macro, align='center',color='red')
    plt.yticks(y_pos, scores_macro.keys())
    plt.title('F1_macro average scores')
    plt.show()

    y_pos = np.arange(len(f1_scores_micro))
    plt.barh(y_pos, f1_scores_micro, align='center',color='yellow')
    plt.yticks(y_pos, scores_micro.keys())
    plt.title('F1_micro average scores')
    plt.show()

    
    y_pos = np.arange(len(f1_scores_weighted))
    plt.barh(y_pos, f1_scores_weighted, align='center',color='green')
    plt.yticks(y_pos, scores_weighted.keys())
    plt.title('F1_weighted average scores')
    plt.show()
    
    
    return


# In[12]:


function(train,train_labels,test,test_labels)


# Ακρίβεια -Precision- ($P$) είναι ο λόγος των true positives ($T_p$) ως προς τον αριθμό των true positives συν τον αριθμό των false positives ($F_p$). $$P = \frac{T_p}{T_p+F_p}$$ Ανάκληση -Recall- ($R$) είναι ο λόγος των true positives ($T_p$) ως προς τον αριθμό των true positives συν τον αριθμό των false negatives ($F_n$). $$R = \frac{T_p}{T_p + F_n}$$ Συχνά χρησιμοποιούμε και το ($F_1$) score, το οποίο είναι ο αρμονικός μέσος της ακρίβειας και της ανάκλησης. $$F1 = 2\frac{P \times R}{P+R}$$ Ιδανικά θέλουμε και υψηλή ακρίβεια και υψηλή ανάκληση, ωστόσο μεταξύ της ακρίβειας και της ανάκλησης υπάρχει γενικά trade-off. Στην οριακή περίπτωση του ταξινομητή που επιστρέφει σταθερά μόνο τη θετική κλάση, η ανάκληση θα είναι 1 ($F_n=0$) αλλά η ακρίβεια θα έχει τη μικρότερη δυνατή τιμή της. Γενικά, κατεβάζοντας το κατώφλι της απόφασης του ταξινομητή, αυξάνουμε την ανάκληση και μειώνουμε την ακρίβεια και αντιστρόφως.
# 
# Στην πράξη και ειδικά σε μη ισορροπημένα datasets χρησιμοποιούμε την ακρίβεια, ανάκληση και το F1 πιο συχνά από την πιστότητα.

# Βλέπουμε ότι η MLP υπερτερεί των άλλων και μάλιστα βρίσκεται σε πολύ καλό σημείο από πλευράς accuracy,γεγονός αναμενόμενο λόγω και της πολυπλοκότητας του συγκεκριμένου ταξινομητή ως προς τη δομή του. Οι dummies από την άλλη πλευρά είναι αναμενόμενο ότι θα κινούνταν σε ιδιαίτερα χαμηλά επίπεδα δεδομένου του πλήθους των διαφορετικών κλάσεων του dataset. 

# # Δ. Βελτιστοποίηση ταξινομητών

# Θα εξετάσουμε μήπως μπορούμε να μειώσουμε τις διαστάσεις του dataset μέσω της επιλογής χαραχτηριστικών που έχουν μηδενικό είτε απειροελάχιστο variance οπότε δεν παίζουν κάποιο ρόλο στην απόφαση για το ποια θα είναι η πρόβλεψή μας.

# In[13]:


from sklearn.feature_selection import VarianceThreshold
# αρχικοποιούμε έναν selector
selector = VarianceThreshold()
# όπως κάναμε και με τους ταξινομητές τον κάνουμε fit στα δεδομένα εκπαίδευσης
train_reduced = selector.fit_transform(train)
mask = selector.get_support()
#print mask

print ('Το παλιό size του train dataset ήταν',train.shape,'. Μετά τη μείωση της διαστατικότητας αυτό είναι',train_reduced.shape)

# εφαρμόζουμε τις αντίστοιχες αλλαγές και στο test set
test_reduced = test[:,mask]
print ('Το παλιό size του test dataset ήταν',test.shape,'. Μετά τη μείωση της διαστατικότητας αυτό είναι',test_reduced.shape)



# Διαπιστώνουμε ότι δεν πετυχαίνει η μείωση της διαστατικότητας μέσω αυτής της μεθόδου οπότε θα στηριχτούμε στην PCA που γίνεται κατά τη βελτιστοποίηση των ταξινομητών.

# Για όλους του Dummy Classifiers δεν έχει νόημα να βρούμε βέλτιστες υπερπαραμέτρους για το pipeline, οπότε κάνουμε χρήση του pipeline με τις default παραμέτρους (δεν θα βρούμε βελτιστες με searchgridcv).

# In[16]:


from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time

# αρχικοποιούμε τους εκτιμητές (μετασχηματιστές και ταξινομητή) χωρείς παραμέτρους
selector = VarianceThreshold()
scaler = preprocessing.StandardScaler()
ros = RandomOverSampler()
pca = PCA()

dc_uni = DummyClassifier(strategy="uniform")
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('dummy', dc_uni)])

start_time = time.time()
pipe.fit(train_dummy,train_labels_dummy)
print("Για τον Dummy Classifier(uniform) : %s seconds" % (time.time() - start_time))

preds = pipe.predict(test_dummy)

#################
print ('Classification report for Dummy Classifier (uniform)')
cr_dummy_uni = classification_report(test_labels, preds)
print (cr_dummy_uni)

scores_micro = {}
scores_macro = {}

scores_micro['Dummy-Uniform']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['Dummy-Uniform']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for Dummy Classifier (uniform)')
print (confusion_matrix(test_labels, preds))

acc_dummy_uni = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_dummy_uni))


# In[17]:


dc_freq = DummyClassifier(strategy="most_frequent")
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('dummy', dc_freq)])

start_time = time.time()
pipe.fit(train_dummy,train_labels_dummy)
print("Για τον Dummy Classifier(Most Frequent) : %s seconds" % (time.time() - start_time))

preds = pipe.predict(test_dummy)

#################
print ('Classification report for Dummy Classifier (most frequent)')
cr_dummy_const_p1 = classification_report(test_labels, preds)
print (cr_dummy_const_p1)

scores_micro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for Dummy Classifier (most frequent)')
print (confusion_matrix(test_labels, preds))

acc_dummy_freq = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_dummy_freq))


# In[19]:


dc_strat = DummyClassifier(strategy="stratified")
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('dummy', dc_strat)])

start_time = time.time()
pipe.fit(train_dummy,train_labels_dummy)
print("Για τον Dummy Classifier(Stratified) : %s seconds" % (time.time() - start_time))

preds = pipe.predict(test_dummy)

#################
print ('Classification report for Dummy Classifier (stratified)')
cr_dummy_strat = classification_report(test_labels, preds)
print (cr_dummy_strat)

scores_micro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['Dummy-Most_Freq']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for Dummy Classifier (stratified)')
print (confusion_matrix(test_labels, preds))

acc_dummy_strat = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_dummy_strat))


# Ουσιαστικά καμία απολύτως βελτίωση για τους dummy classifiers, ο αριθμός των κλάσεων δεν τους επέτρεξε να λειτουργήσουν.

# Όπως στους dummy, ετσι και στον Naive Bayes Classifier θα δοκιμάσουμε τις default τιμές για το pipeline.

# In[20]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
# κάνουμε εκπαίδευση (fit) δηλαδή ουσιαστικά υπολογίζουμε μέση τιμή και διακύμανση για όλα τα χαρακτηριστικά και κλάσεις στο training set
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('gnb', gnb)])

start_time = time.time()
pipe.fit(train_dummy,train_labels_dummy)
print("Για τον Gaussian Naive Bayes: %s seconds" % (time.time() - start_time))

preds = pipe.predict(test)

#################
print ('Classification report for Gaussian Naive Bayes Classifier')
cr_gnb = classification_report(test_labels, preds)
print (cr_gnb)

scores_micro['GNB']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['GNB']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for Gaussian Naive Bayes Classifier')
print (confusion_matrix(test_labels, preds))

acc_gnb = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_gnb))


# Χαμηλά αποτελέσματα και για τον gnbc. Συνεχίζουμε με τον kNN τον οποίο θα επιχειρήσουμε να βελτιστοποιήσουμε μέσω gridsearch.

# In[22]:


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_jobs=-1) # η παράμετρος n_jobs = 1 χρησιμοποιεί όλους τους πυρήνες του υπολογιστή
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('kNN', clf)])

start_time = time.time()
pipe.fit(train, train_labels)
print("Για τον non-optimized kNN : %s seconds" % (time.time() - start_time))

preds = pipe.predict(test)


#################
print ('Classification report for non-optimized kNN')
cr_knn_no = classification_report(test_labels, preds)
print (cr_knn_no)

scores_micro['kNN-non-opt']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['kNN-non-opt']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for non-optimized kNN')
print (confusion_matrix(test_labels, preds))

acc_knn_no = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_knn_no))


# Επιχειρούμε βελτιστοποίηση. Ξεκινάμε υπολογίζοντας τη διασπορά του δείγματος καθώς ενδέχεται να παίξει ρόλο στην επιλογή του v_threshold.

# In[23]:


train_variance = train.var(axis=0)
print(train_variance)
print(np.max(train_variance))


# Συμπεραίνουμε ότι λογικά θα χρειαστούμε v_threshold πολύ κοντά στο 0.

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

### PREPROCESSING
vthreshold = [0,0.1]
n_components = [35,50,65,80]

### kNN specific
k_neighbors = [1,5,7,9,11,17]
metric =['euclidean']
weights =['uniform','distance']
verbose=10

clf = neighbors.KNeighborsClassifier(n_jobs=-1) # η παράμετρος n_jobs = -1 χρησιμοποιεί όλους τους πυρήνες του υπολογιστή
pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('kNN', clf)])

estimator_macro = GridSearchCV(pipe, dict(selector__threshold=vthreshold, pca__n_components=n_components,kNN__n_neighbors=k_neighbors,kNN__weights=weights,kNN__metric=metric),verbose=10,cv=5,scoring='f1_macro', n_jobs=-1)

start_time = time.time()
estimator_macro.fit(train, train_labels)
print("Για τον optimized kNN : %s seconds" % (time.time() - start_time))

preds = estimator_macro.predict(test)

#estimator_weighted = GridSearchCV(pipe, dict(selector__threshold=vthreshold, pca__n_components=n_components,kNN__n_neighbors=k_neighbors,kNN__weights=weights,kNN__metric=metric), cv=5,scoring='f1_weighted', n_jobs=-1)
#estimator_weighted.fit(train, train_labels)
#preds_weighted = estimator_weighted.predict(test)

#################
print ('Classification report for optimized kNN')
cr_knn_opt = classification_report(test_labels, preds)
print (cr_knn_opt)

scores_micro['kNN-opt']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['kNN-opt']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for optimized kNN')
print (confusion_matrix(test_labels, preds))

acc_knn_opt = 100*accuracy_score(test_labels,preds)
print ('\nAccuracy percentage of this classifier is %.3f %%' % (acc_knn_opt))

# print best estimator configuration
print ('\nFor kNN the optimal configuration is :')
print (estimator_macro.best_estimator_)


# Φαίνεται να υπάρχει βελτίωση της τάξης του 5 με 6%, αρκετά ικανοποιητικός αριθμός. Έγιναν δοκιμές για διαφορετικές τιμές διαφόρων παραμέτρων αλλά αυτός φαίνεται να είναι ο αποδοτικότερος συνδυασμός, αν και κάπως χρονοβόρος.

# Επόμενο βήμα η απόπειρα βελτιστοποίησης του mlp classifier.

# In[26]:


clf = MLPClassifier()

pipe = Pipeline(steps=[('selector', selector), ('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', clf)])

### PREPROCESSING
vthreshold = [0]
n_components = [600]#,50,200,300,500,600]

### MLP specific
so = ['lbfgs']
al = [10**float(c) for c in np.arange(-7,-6)]
hd = [45]#, 25,45]
mi = [900]#,400,500,600,1000,800]
lr = ['constant','invscaling','adaptive']
ac = ['identity','logistic','tanh','relu']

estimator_macro = GridSearchCV(pipe, dict(selector__threshold=vthreshold, pca__n_components=n_components,mlp__solver=so, mlp__alpha=al,mlp__hidden_layer_sizes=hd,mlp__max_iter=mi,mlp__learning_rate=lr,mlp__activation=ac),cv=5,scoring='f1_macro', n_jobs=-1)

start_time = time.time()
estimator_macro.fit(train, train_labels)
print("Για τον optimized MLP : %s seconds" % (time.time() - start_time))

preds = estimator_macro.predict(test)

#estimator_weighted = GridSearchCV(pipe, dict(selector__threshold=vthreshold, pca__n_components=n_components,mlp__solver=so, mlp__alpha=al,mlp__hidden_layer_sizes=hd,mlp__max_iter=mi), cv=5,scoring='f1_weighted', n_jobs=-1)
#estimator_weighted.fit(train, train_labels)
#preds_weighted = estimator_weighted.predict(test)

#################
print ('Classification report for optimized MLP')
cr_mlp_opt = classification_report(test_labels, preds)
print (cr_mlp_opt)

scores_micro['MLP-opt']=precision_recall_fscore_support(test_labels,preds,average='micro')
scores_macro['MLP-opt']=precision_recall_fscore_support(test_labels,preds,average='macro')

print ('Confusion Matrix for optimized MLP')
print (confusion_matrix(test_labels, preds))

acc_mlp_opt = 100*accuracy_score(test_labels,preds)
print ('Accuracy percentage of this classifier is %.3f %%' % (acc_mlp_opt))

print ('\nFor MLP the optimal configuration is :')
print (estimator_macro.best_estimator_)


# Ο mlp ηταν αρκετά ψηλά σε accuracy και δεν πετύχαμε καμία βελτιστοποίηση, πιθανώς λόγω τυχαιότητας να προέρχεται η μικρή διαφορά που έχει πριν και μετά τη βελτιστοποίηση. Ακολουθούν τα διαγράμματα που ζητούνται βάσει των f1_micro και f1_macro, καθώς και πίνακας απόδοσης των ταξινομητών πριν και μετά τη βελτιστοποίηση.

# In[28]:


# Κάνουμε import την matplotplib
import matplotlib.pyplot as plt

f1_scores_micro = [item[2] for item in scores_micro.values()]
f1_scores_macro = [item[2] for item in scores_macro.values()]

y_pos = np.arange(len(f1_scores_micro))
plt.barh(y_pos, f1_scores_micro, align='center',color='red')
plt.yticks(y_pos, scores_micro.keys())
plt.title('F1_micro average scores')
plt.show()

y_pos = np.arange(len(f1_scores_macro))
plt.barh(y_pos, f1_scores_macro, align='center',color='green')
plt.yticks(y_pos, scores_macro.keys())
plt.title('F1_macro average scores')
plt.show()


# Παρακάτω παρουσιάζονται οι πίνακες με τα συγκριτικά αποτελέσματα της απόδοσης των ταξινομητών πριν και μετά τη βελτιστοποίηση (μεταβολή του accuracy).

# In[30]:


x= {'%DummyUni': [4.231-4.188], '%DummyMostFreq': [3.590-2.949],'%DummyStrat': [3.889-3.761],'%GNBC':[81.325-84.957],'%kNN':[90.128-85.769],'%MLP':[94.744- 95.812]}
df1 = pd.DataFrame(data=x)
df1


# Συμπερασματικά, βλέπουμε ότι σημαντική βελτίωση είχαμε στον ταξινομητή kNN ενώ ο GNBC δεν βελτιστοποιήθηκε αλλά αντιθέτως μειώθηκε το accuracy του. Συνολικά, όπως φαίνεται και από τα f1_micro και f1_macro scores ο πιο αξιόπιστος ταξινομητής για το dataset μας είναι ο MLP ο οποίος, παρ΄όλο που δεν πετύχαμε κάποια βελτίωση κινείται σε αρκετά υψηλά επίπεδα. Μικρές διαφορές πριν και μετά τη βελτιστοποίηση είναι πιθανό να οφείλονται και σε ένα παράγοντα τυχαιότητας που εμπεριέχει ο υπολογισμός των βέλτιστων υπερπαραμέτρων αλλά και η εκτέλεση των ταξινομητών.

# In[ ]:




