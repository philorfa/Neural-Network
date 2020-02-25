#!/usr/bin/env python
# coding: utf-8

# # A.           Στοιχεία Ομάδας
# 
# &gt; ### ΟΜΑΔΑ A5
# 
# &gt; Αριστείδης Μπακούρος 03113138
# 
# &gt; Ορφανουδάκης Φίλιππος 03113140
# 

# # B.  Εισαγωγή του dataset

# Αρχικά κάνουμε upgrade στις βιβλοθήκες που θα χρειαστούμε

# In[1]:


get_ipython().system(u'pip install --upgrade pip #upgrade pip package installer')
get_ipython().system(u'pip install scikit-learn --upgrade #upgrade scikit-learn package')
get_ipython().system(u'pip install numpy --upgrade #upgrade numpy package')
get_ipython().system(u'pip install pandas --upgrade #upgrade pandas package')


# Ύστερα αρχιζουμε να μελετάμε το dataset μας το οποίο αντιστοιχεί στο S06 και ονομάζεται Japanese Credit Screening , αλλά καθώς ανοίγουμε τις πληροφορίες του δηλαδή το Data folder και παρατηρήσουμε με ποιό dataset θα δουλέψουμε καταλήγουμε ότι ονομάζεται Credit Approval.
# 
# Η πληροφορία που μεταφέρει αυτο το dataset με λίγα λόγια ειναι η εξής , αιτήσεις για δάνεια και αν τελικά έγιναν δεκτές ή όχι.
# Πιο συγκεκριμένα έχουμε 690 αιτήσεις - instances κάθε μια διαθέτει 15 κρυπτογραφημένα χαρακτηριστικά - attributes (για προστασία των προσωπικών δεδομένων) για τον αιτουντα και ένα έξτρα χαρακτηριστικό το οποίο εκφράζει αν τελικά πήρε έγκριση (+) ή όχι (-).
# Αυτό μας οδηγεί και στο συμπέρασμα ότι έχουμε 2 κλάσεις στο dataset μας.
# 
# Άλλες πληροφορίες που παίρνουμε είναι ότι : 
#    * Εχουμε 37 περιπτώσεις με απουσιάζουσες τιμές το οποίο αντιστοιχεί στο 5% του dataset μας
#    * Το dataset μας είναι ισορροπημένο καθώς έχουμε συχνότητα εμφάνισης   + -&gt; 307 (44.5%) και   - -&gt; 383 (55.5%)
#    * Διαθέτουμε attributes κατηγορικά, μη διατεταγμένα αλλα και διατεταγμένα οπως φαινεται παρακάτω :
#     
#  A1:	b, a.
#  
#  A2:	continuous.
#  
#  A3:	continuous.
#  
#  A4:	u, y, l, t.
#  
#  A5:	g, p, gg.
#  
#  A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
#  
#  A7:	v, h, bb, j, n, z, dd, ff, o.
#  
#  A8:	continuous.
#  
#  A9:	t, f.
#  
#  A10:	t, f.
#  
#  A11:	continuous.
#  
#  A12:	t, f.
#  
#  A13:	g, p, s.
#  
#  A14:	continuous.
#  
#  A15:	continuous.

# Για να εισάγουμε το dataset μας κατεβάζουμε το αρχείο crx.data το ανεβάζουμε στο ιδιο directory με αυτό το notebook

# In[2]:


get_ipython().system(u'ls')


# Διαβάζουμε το dataset μας και επειδή έχουμε ενημερωθεί για τις missing values και οτι μεταφράζονται με ερωτηματικό ? στο dataset , τις αναγνωρίζουμε κατα την ανάγωση .

# In[3]:


import pandas as pd

df = pd.read_csv("crx.data",na_values = ["?"])
df.head()


# Παρατηρούμε ότι η πρώτη γραμμή δεν ειναι επικεφαλίδα και δεν δίνει κάποια ονομασία στα attributes , αλλά αποτελεί ένα instance οπότε το χρειαζόμαστε . Επίσης βλέπουμε ότι η αρίθμηση γίνεται αριστερά των instances.

# In[4]:


df = pd.read_csv("crx.data", header=None,na_values = ["?"])
df.head()


# In[5]:


print("Έχουμε", len(df), "δείγματα")
print("Με το καθένα να έχει ", df.shape[1], "χαρακτηριστικά")


# Παρακάτω φαινονται όλα τα instances που διαθέτουν 1 ή παραπανω missing value

# In[6]:


null_columns=df.columns[df.isnull().any()]
print(df[df.isnull().any(axis=1)][null_columns].head(37))


# Και αναλυτικά πόσα attributes έχουν missing values 

# In[7]:


import numpy as np

df_1 = pd.read_csv("crx.data",header=None,na_values = ["?"])
null_columns=df_1.columns[df_1.isnull().any()]
df_1[null_columns].isnull().sum()


# In[8]:


print ('Οι απουσιάζουσες τιμές είναι',df_1[null_columns].isnull().sum().sum())


# Έχω 67 missing values το οποίο αντιστοιχεί σε 37 attributes δηλαδή το 5% του συνολικού μας dataset , αποτελεί ένα μικρό ποσοστό το οποίο αποφασίζουμε να το χειριστούμε με διαγραφή. 

# In[9]:


df1=df.dropna()
df1


# Πλέον έχουμε 652 instances όπως φαινετα, και αποφασσίζουμε να δώσουμε όνομα σε κάθε χαρακτηριστικό

# In[10]:


df1.columns = [
  'A1',
  'A2',
    'A3',
    'A4',
    'A5',
    'A6',
    'A7',
    'A8',
    'A9',
    'A10',
    'A11',
    'A12',
    'A13',
    'A14',
    'A15',
    'GRANTED'
]


# In[11]:


df1


# Όπως φαίνεται και από τις πληροφορίες του dataset αλλα και απο το ίδιο το dataset έχουμε ορισμένα μη διατεγμένα χαρακτηριστικά, ο τρόπος χειρισμου τους είναι η αύξηση των στηλών και της διαστατικότητας ετσι ώστε να περιγράφονται απόλυτα τα χαρακτηριστικά .
# Αυτο το επιτυγχάνουμε με την get_dummies

# In[12]:


pd.get_dummies(df1,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])


# Έτσι καταλήγουμε σε 47 στήλες
# 
# Στη συνέχεια θέλουμε να αντιμετωπίσουμε την binary class μας , και αυτο που κάνουμε είναι η αραιή-sparse αναπαράσταση με 0 όσες αιτήσεις απορρίφθηκαν και 1 όσες αιτήσεις έγιναν δεκτές.
# 

# In[13]:


df2=pd.get_dummies(df1,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
df3=df2
df3.loc[df3.GRANTED != '+', 'GRANTED'] = 0
df3.loc[df3.GRANTED == '+', 'GRANTED'] = 1


# In[14]:


df3


# Στη συνέχεια παρατηρύμε οτι η στήλη GRANTED ουσιαστικά αποτελεί την έξοδο μας , βρίσκεται στη μέση και μας δυσκολευει , για διευκόλυνση ορίζουμε ξανά το dataset και την βάζουμε στο τέλος

# In[15]:


cols = list(df3.columns.values) 
cols.pop(cols.index('GRANTED')) 
df4 = df3[cols+['GRANTED']] 


# In[16]:


df4


# In[17]:


list(df4)


# Χωρίζουμε το dataset μας σε labels και features

# In[18]:


label_names = ["Not Granted","Granted"]
labels_df = df4.iloc[:,[46]]
##feature_names = list(df4)
features_df = df4.iloc[:,:45]


# Μετά την διαγραφή των instances με missing values πάμε να ελέγξουμε πάλι την ισορροπία του δείγματος μας

# In[19]:


labels = labels_df.values.flatten()
mapping_classes = {0:"Not_Granted",1:"Granted"}
discrete_classes = list(set(labels))
for i in discrete_classes:
    print ('Εμφανίζεται ποσοστό δειγμάτων %.2f %% για την κλάση %s' % (100.0*sum(labels==i)/len(labels),mapping_classes[i]))


# In[20]:


labels_df_t = df4['GRANTED']
labels_df_t = list(map(lambda x : x, labels_df_t))


# In[21]:


features_df_t=np.asarray(features_df)


# Αφου μετατρέψαμε κατάλληλα το dataset μας και χωρίσαμε σε features και labels είμαστε έτοιμοι να διαχωρίσουμε σε train και test set 

# In[22]:


from sklearn.model_selection import train_test_split

# Split our data
train, test, train_labels, test_labels = train_test_split(features_df_t, labels_df_t, test_size=0.2,random_state=1)


# # Γ. Baseline classification

# Θα χρησιμοποιήσουμε τους ταξινομητές kNN και dummy χωρίς καμια βελτιστοποίηση ούτε του dataset ούτε των υπερπαραμέτρων της kNN.
# Πιο συγκεκριμένα στην dummy θα δουλέψουμε όλες τις τακτικές ταξινόμησης που διαθέτει δηλαδή : 
# *                                                                                                  uniform
# *                                                                                                  constant 0,1
# *                                                                                                  most frequent
# *                                                                                                  stratified

# In[23]:


from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


# In[111]:


def function(train_in,train_labels_in,test_in,test_labels_in,fmac,fmic,fwei):
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
    
    fmac[0]=f1_score(test_labels_in,preds,average='macro')
    fmic[0]=f1_score(test_labels_in,preds,average='micro')
    fwei[0]=f1_score(test_labels_in,preds,average='weighted')
    
    
    dc_constant_1 = DummyClassifier(strategy="constant", constant=1)
    model = dc_constant_1.fit(train_in, train_labels_in)
    preds = dc_constant_1.predict(test_in)
    credit_accuracy['constant 1'] = dc_constant_1.score(test_in, test_labels_in)


#################
    print ('Classification report for Dummy Classifier (constant-1)')
    cr_dummy_const1 = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_const1)

    scores_weighted['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='micro')


    print ('Confusion Matrix for Dummy Classifier (constant-1)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_const1 = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_const1))
    
    fmac[1]=f1_score(test_labels_in,preds,average='macro')
    fmic[1]=f1_score(test_labels_in,preds,average='micro')
    fwei[1]=f1_score(test_labels_in,preds,average='weighted')
    
    dc_constant_0 = DummyClassifier(strategy="constant", constant=0)
    model = dc_constant_0.fit(train_in, train_labels_in)
    preds = dc_constant_0.predict(test_in)
    credit_accuracy['constant 0'] = dc_constant_0.score(test_in, test_labels_in)

#################
    print ('Classification report for Dummy Classifier (constant-0)')
    cr_dummy_const0 = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_const0)

    scores_weighted['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='micro')

    print ('Confusion Matrix for Dummy Classifier (constant-0)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_const0 = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_const0))
    
    fmac[2]=f1_score(test_labels_in,preds,average='macro')
    fmic[2]=f1_score(test_labels_in,preds,average='micro')
    fwei[2]=f1_score(test_labels_in,preds,average='weighted')
    
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
    
    fmac[3]=f1_score(test_labels_in,preds,average='macro')
    fmic[3]=f1_score(test_labels_in,preds,average='micro')
    fwei[3]=f1_score(test_labels_in,preds,average='weighted')
    
    
    
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
    
    fmac[4]=f1_score(test_labels_in,preds,average='macro')
    fmic[4]=f1_score(test_labels_in,preds,average='micro')
    fwei[4]=f1_score(test_labels_in,preds,average='weighted')
    
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

    print ('Confusion Matrix for non-optimized kNN')
    print (confusion_matrix(test_labels_in, knn_preds))

    acc_knn_no = 100*accuracy_score(test_labels_in,knn_preds)
    print ('\nAccuracy percentage of this classifier is %.3f %%\n' % (acc_knn_no))
    
    fmac[5]=f1_score(test_labels_in,knn_preds,average='macro')
    fmic[5]=f1_score(test_labels_in,knn_preds,average='micro')
    fwei[5]=f1_score(test_labels_in,knn_preds,average='weighted')
    
    
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


# Έχουμε κατασκευάσει μια συνάρτηση η οποία δέχεται το train, train_labels , test , test labels και εκπαιδεύει τον dummy για κάθε τακτική του και τον kNN που έχει default την υπερπαραπετρο n_neighbors.
# 
# Για κάθε ταξινομήτη εμφανίζεται το classification report του , το confusion matrix του , και το ποσοστό ακρίβειας του.
# Στο τέλος εμφανίζονται plots που γίνεται συγκριση κάθε τακτικής ταξινομητη για f1_macro avg, f1_micro avg και f1_weighted avg

# In[112]:


fwei=[0,0,0,0,0,0]
fmac=[0,0,0,0,0,0]
fmic=[0,0,0,0,0,0]
function(train,train_labels,test,test_labels,fwei,fmac,fmic)


# Ακρίβεια -Precision- ($P$) είναι ο λόγος των true positives ($T_p$) ως προς τον αριθμό των true positives συν τον αριθμό των false positives ($F_p$).
# $$P = \frac{T_p}{T_p+F_p}$$
# Ανάκληση -Recall- ($R$) είναι ο λόγος των true positives ($T_p$) ως προς τον αριθμό των true positives συν τον αριθμό των false negatives ($F_n$).
# $$R = \frac{T_p}{T_p + F_n}$$
# Συχνά χρησιμοποιούμε και το ($F_1$) score, το οποίο είναι ο αρμονικός μέσος της ακρίβειας και της ανάκλησης.
# $$F1 = 2\frac{P \times R}{P+R}$$
# Ιδανικά θέλουμε και υψηλή ακρίβεια και υψηλή ανάκληση, ωστόσο μεταξύ της ακρίβειας και της ανάκλησης υπάρχει γενικά trade-off. Στην οριακή περίπτωση του ταξινομητή που επιστρέφει σταθερά μόνο τη θετική κλάση για παράδειγμα, η ανάκληση θα είναι 1 αλλά η ακρίβεια θα έχει τη μικρότερη δυνατή τιμή της. Γενικά, κατεβάζοντας το κατώφλι της απόφασης του ταξινομητή, αυξάνουμε την ανάκληση και μειώνουμε την ακρίβεια και αντιστρόφως. 
# 
# 
# 
# ### Παρατηρούμε την εμφανή υπεροχή σε όλα τα scores του kNN , αν αγνοήσουμε την constant τακτική του dummy που είναι λογικό να έχει ανεβασμένα νούμερα στα αντίστοιχα classes.

# # Δ. Βελτιστοποίηση ταξινομητών

# Σε αυτή τη φάση πάμε να βρούμε τη βέλτιστη αρχιτεκτονική για κάθε ταξινομητή.
# Επειδή ο dummy δεν επιδέχεται βελτίωση θα ασχοληθούμε καθαρά με τον kNN.
# 
# Πιο συγκεκριμένα βρίκσουμε τις διαθέσιμες βελτιστοποιήσεις για το dataset μας οι οποιές ειναι : VarianceThreshold,imbalanced-learn MinMaxScaler,StandardScaler και PCA .
# 
# Φτιάχνουμε συναρτήσεις που να δέχονται ένα συγκρκιμένο train , test set , και εφαρμόζουν τις βελτιστοποιήσεις πάνω τους.
# 
# Για τις εξόδους των παραπάνων συναρτήσεων εφαρμόζουμε cross validation η οποία θα μας δώσει τη βέλτιστη υπερπαράμερτο n_neighbors για την εφαρμογή του ταξινομητη kNN 
# 
# Ύστερα έχοντας αυτα τα δεδομένα εκπαιδεύω τον kNN και βγάζω τα αποτελεσματα μου.

# Αρχικά βλέπουμε αν ειναι ισορροπημενο το train,test set μας , ετσι ωστε αν χρειαστει να κόψω κάποια instances μέσω της imbalanced

# In[26]:


print('το αρχικο train set εχει ',train_labels.count(0),'δειγματα κατηγοριας 0 - not granted')


# In[27]:


print('το αρχικο train set εχει ',train_labels.count(1),'δειγματα κατηγοριας 1 - granted')


# Βλέπουμε όπως ειναι αναμενόμενο ότι έχει κρατήσει την ισορροπία του και το train,test set όπως και το αρχικό, επομένως δεν θα χρειαστει αυτή η βελτιστοποίηση ούτε και στη πορεία καθώς όλες οι υπολοιπες βελτιστοποιήσεις δεν επηρεάζουν την ισορροπία

# Πάμε να συνεχίσουμε με τις υπόλοιπες , το σκεπτικό με το οποίο εργαστήκαμε είναι να δημιουργήσουμε όλες τους πιθανους συνδυασμούς των μετασχηματιστών και στη συνέχεια να περάσουμε το set μας στον cross validation και τέλος στον kNN .
# 
# To cross validation όπως και κάθε συνδυασμός έχει υλοποιηθεί 2 φορές , 1 για την f1_macro και 1 για την f1_micro

# In[28]:


from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score
import time


# In[29]:


def variance(train,test,thresh):
    selector = VarianceThreshold(threshold=thresh)
    train_reduced = selector.fit_transform(train)
    mask = selector.get_support()
    test_reduced = test[:,mask]
    return train_reduced,test_reduced


# In[30]:


def min_max(train,test):
    min_max_scaler = preprocessing.MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(train)
    test_scaled = min_max_scaler.transform(test)
    return train_scaled,test_scaled


# In[31]:


def scaled_Standard(train,test):

    scaler = preprocessing.StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled,test_scaled


# In[32]:


def pca(n,train,test):

    pcaa = PCA(n_components=n)


    trainPCA =  pcaa.fit_transform(train)
    testPCA = pcaa.transform(test)


    return trainPCA , testPCA


# In[33]:


def cross_macro(train):

    myList = list(range(1,50))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))
    cv_scores = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train, train_labels, cv=10, scoring='f1_macro')
        cv_scores.append(scores.mean())
    mean_error = [1 - x for x in cv_scores]
    optimal_k = neighbors[mean_error.index(min(mean_error))]
    return optimal_k


# In[34]:


def cross_micro(train):

    myList = list(range(1,50))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))
    cv_scores = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train, train_labels, cv=10, scoring='f1_micro')
        cv_scores.append(scores.mean())
    mean_error = [1 - x for x in cv_scores]
    optimal_k = neighbors[mean_error.index(min(mean_error))]
    return optimal_k


# In[35]:


def acc_macro(train_in,test_in,num):
    
    knn = KNeighborsClassifier(n_neighbors = num)
    knn.fit(train_in, train_labels)
    knn_preds = knn.predict(test_in)
    scores_weighted = {}
    scores_macro = {}
    scores_micro = {}
    return f1_score(test_labels,knn_preds,average='macro')


# In[36]:


def acc_micro(train_in,test_in,num):
    
    knn = KNeighborsClassifier(n_neighbors = num)
    knn.fit(train_in, train_labels)
    knn_preds = knn.predict(test_in)
    scores_weighted = {}
    scores_macro = {}
    scores_micro = {}
    return f1_score(test_labels,knn_preds,average='micro')


# In[37]:


def minim(a,b):
    if a <= b:
        c = a - 1
    else:
        c = b
    return c


# In[38]:


def knn_macro(maxim,nei,pc,th):
    k=cross_macro(train)
    fin=acc_macro(train,test,k)
    maxim[0]=fin
    nei[0]=k
    return


# In[39]:


def knn_micro(maxim,nei,pc,th):
    k=cross_micro(train)
    fin=acc_micro(train,test,k)
    maxim[0]=fin
    nei[0]=k
    return


# In[40]:


def minmax_knn_macro(maxim,nei,pc,th):
    train_1,test_1=min_max(train,test)
    k=cross_macro(train_1)
    fin=acc_macro(train_1,test_1,k)
    maxim[1]=fin
    nei[1]=k
    return


# In[41]:


def minmax_knn_micro(maxim,nei,pc,th):
    train_1,test_1=min_max(train,test)
    k=cross_micro(train_1)
    fin=acc_micro(train_1,test_1,k)
    maxim[1]=fin
    nei[1]=k
    return


# In[42]:


def standard_knn_macro(maxim,nei,pc,th):
    train_1,test_1=scaled_Standard(train,test)
    k=cross_macro(train_1)
    fin=acc_macro(train_1,test_1,k)
    maxim[2]=fin
    nei[2]=k
    return


# In[43]:


def standard_knn_micro(maxim,nei,pc,th):
    train_1,test_1=scaled_Standard(train,test)
    k=cross_micro(train_1)
    fin=acc_micro(train_1,test_1,k)
    maxim[2]=fin
    nei[2]=k
    return


# In[44]:


def pca_knn_macro(maxim,nei,pc,th):
    maxim[3]=0
    maxn_pc = minim(train.shape[0], train.shape[1])
    for y in range(1,maxn_pc,1):
        train_1,test_1 = pca(y,train,test)
        k=cross_macro(train_1)
        fin=acc_macro(train_1,test_1,k)
        if fin>maxim[3]:
            maxim[3]=fin
            pc[3]=y
            nei[3]=k
    return


# In[45]:


def pca_knn_micro(maxim,nei,pc,th):
    maxim[3]=0
    maxn_pc = minim(train.shape[0], train.shape[1])
    for y in range(1,maxn_pc,1):
        train_1,test_1 = pca(y,train,test)
        k=cross_micro(train_1)
        fin=acc_micro(train_1,test_1,k)
        if fin>maxim[3]:
            maxim[3]=fin
            pc[3]=y
            nei[3]=k
    return


# In[46]:


def minmax_pca_knn_macro(maxim,nei,pc,th):
    maxim[4]=0
    train_1,test_1=min_max(train,test)
    maxn_pc = minim(train_1.shape[0], train_1.shape[1])
    for y in range(1,maxn_pc,1):
        train_2,test_2 = pca(y,train_1,test_1)
        k=cross_macro(train_2)
        fin=acc_macro(train_2,test_2,k)
        if fin>maxim[4]:
            maxim[4]=fin
            pc[4]=y
            nei[4]=k
    return


# In[47]:


def minmax_pca_knn_micro(maxim,nei,pc,th):
    maxim[4]=0
    train_1,test_1=min_max(train,test)
    maxn_pc = minim(train_1.shape[0], train_1.shape[1])
    for y in range(1,maxn_pc,1):
        train_2,test_2 = pca(y,train_1,test_1)
        k=cross_micro(train_2)
        fin=acc_micro(train_2,test_2,k)
        if fin>maxim[4]:
            maxim[4]=fin
            pc[4]=y
            nei[4]=k
    return


# In[48]:


def standard_pca_knn_macro(maxim,nei,pc,th):
    maxim[5]=0
    train_1,test_1=scaled_Standard(train,test)
    maxn_pc = minim(train_1.shape[0], train_1.shape[1])
    for y in range(1,maxn_pc,1):
        train_2,test_2 = pca(y,train_1,test_1)
        k=cross_macro(train_2)
        fin=acc_macro(train_2,test_2,k)
        if fin>maxim[5]:
            maxim[5]=fin
            pc[5]=y
            nei[5]=k
    return


# In[49]:


def standard_pca_knn_micro(maxim,nei,pc,th):
    maxim[5]=0
    train_1,test_1=scaled_Standard(train,test)
    maxn_pc = minim(train_1.shape[0], train_1.shape[1])
    for y in range(1,maxn_pc,1):
        train_2,test_2 = pca(y,train_1,test_1)
        k=cross_micro(train_2)
        fin=acc_micro(train_2,test_2,k)
        if fin>maxim[5]:
            maxim[5]=fin
            pc[5]=y
            nei[5]=k
    return


# In[50]:


def thres_knn_macro(maxim,nei,pc,th,a,b,c):   
    maxim[6]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        k=cross_macro(train_1)
        fin=acc_macro(train_1,test_1,k)
        if fin>maxim[6]:
            maxim[6]=fin
            th[6]=x
            nei[6]=k
    return


# In[51]:


def thres_knn_micro(maxim,nei,pc,th,a,b,c):   
    maxim[6]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        k=cross_micro(train_1)
        fin=acc_micro(train_1,test_1,k)
        if fin>maxim[6]:
            maxim[6]=fin
            th[6]=x
            nei[6]=k
    return


# In[52]:


def thres_minmax_knn_macro(maxim,nei,pc,th,a,b,c):   
    maxim[7]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=min_max(train_1,test_1)
        k=cross_macro(train_1)
        fin=acc_macro(train_1,test_1,k)
        if fin>maxim[7]:
            maxim[7]=fin
            th[7]=x
            nei[7]=k
    return


# In[53]:


def thres_minmax_knn_micro(maxim,nei,pc,th,a,b,c):   
    maxim[7]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=min_max(train_1,test_1)
        k=cross_micro(train_1)
        fin=acc_micro(train_1,test_1,k)
        if fin>maxim[7]:
            maxim[7]=fin
            th[7]=x
            nei[7]=k
    return


# In[54]:


def thres_standard_knn_macro(maxim,nei,pc,th,a,b,c) :   
    maxim[8]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=scaled_Standard(train_1,test_1)
        k=cross_macro(train_1)
        fin=acc_macro(train_1,test_1,k)
        if fin>maxim[8]:
            maxim[8]=fin
            th[8]=x
            nei[8]=k
    return


# In[55]:


def thres_standard_knn_micro(maxim,nei,pc,th,a,b,c) :   
    maxim[8]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=scaled_Standard(train_1,test_1)
        k=cross_micro(train_1)
        fin=acc_micro(train_1,test_1,k)
        if fin>maxim[8]:
            maxim[8]=fin
            th[8]=x
            nei[8]=k
    return


# In[56]:


def thres_minmax_pca_knn_macro(maxim,nei,pc,th,a,b,c):    
    maxim[9]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=min_max(train_1,test_1)
        maxn_pc = minim(train_1.shape[0], train_1.shape[1])
        for y in range(1,maxn_pc,1):
            train_2,test_2 = pca(y,train_1,test_1)
            k=cross_macro(train_2)
            fin=acc_macro(train_2,test_2,k)
            if fin>maxim[9]:
                maxim[9]=fin
                th[9]=x
                pc[9]=y
                nei[9]=k
    return


# In[57]:


def thres_minmax_pca_knn_micro(maxim,nei,pc,th,a,b,c):    
    maxim[9]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=min_max(train_1,test_1)
        maxn_pc = minim(train_1.shape[0], train_1.shape[1])
        for y in range(1,maxn_pc,1):
            train_2,test_2 = pca(y,train_1,test_1)
            k=cross_micro(train_2)
            fin=acc_micro(train_2,test_2,k)
            if fin>maxim[9]:
                maxim[9]=fin
                th[9]=x
                pc[9]=y
                nei[9]=k
    return


# In[58]:


def thres_standard_pca_knn_macro(maxim,nei,pc,th,a,b,c):    
    maxim[10]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=scaled_Standard(train_1,test_1)
        maxn_pc = minim(train_1.shape[0], train_1.shape[1])
        for y in range(1,maxn_pc,1):
            train_2,test_2 = pca(y,train_1,test_1)
            k=cross_macro(train_2)
            fin=acc_macro(train_2,test_2,k)
            if fin>maxim[10]:
                maxim[10]=fin
                th[10]=x
                pc[10]=y
                nei[10]=k
    return 


# In[59]:


def thres_standard_pca_knn_micro(maxim,nei,pc,th,a,b,c):    
    maxim[10]=0
    for x in np.arange(a,b,c):
        train_1,test_1=variance(train,test,x)
        train_1,test_1=scaled_Standard(train_1,test_1)
        maxn_pc = minim(train_1.shape[0], train_1.shape[1])
        for y in range(1,maxn_pc,1):
            train_2,test_2 = pca(y,train_1,test_1)
            k=cross_micro(train_2)
            fin=acc_micro(train_2,test_2,k)
            if fin>maxim[10]:
                maxim[10]=fin
                th[10]=x
                pc[10]=y
                nei[10]=k
    return 


# In[60]:


def pipe_grid_macro(a,b,c):
    maxim=[0,0,0,0,0,0,0,0,0,0,0]
    nei=[0,0,0,0,0,0,0,0,0,0,0]
    pc=[0,0,0,0,0,0,0,0,0,0,0]
    th=[0,0,0,0,0,0,0,0,0,0,0]
    mx=0
    thes=0
    knn_macro(maxim,nei,pc,th)
    print('9%')
    minmax_knn_macro(maxim,nei,pc,th)
    print('18%')
    standard_knn_macro(maxim,nei,pc,th)
    print('27%')
    pca_knn_macro(maxim,nei,pc,th)
    print('36%')
    minmax_pca_knn_macro(maxim,nei,pc,th)
    print('45%')
    standard_pca_knn_macro(maxim,nei,pc,th)
    print('54%')
    thres_knn_macro(maxim,nei,pc,th,a,b,c)
    print('63%')
    thres_minmax_knn_macro(maxim,nei,pc,th,a,b,c)
    print('72%')
    thres_standard_knn_macro(maxim,nei,pc,th,a,b,c)
    print('81%')
    thres_minmax_pca_knn_macro(maxim,nei,pc,th,a,b,c)
    print('90%')
    thres_standard_pca_knn_macro(maxim,nei,pc,th,a,b,c)
    print('99%')
    print('\n')
    print('\n')
    print('\n')
    print("!!!!!!!!!")
    print("H βέλτιστη αρχιτεκτονική για f1_macro ειναι η εξής:\n")
    for i in range(11):
        if(maxim[i]>mx):
            thes=i
            mx=maxim[i]
    if thes==0:
        print('kNN με cross validation έχω f_macro=',maxim[0],'για k_neighbors=',nei[0],'\n')
    elif thes==1:
        print('min_max και kNN με cross validation έχω f_macro=',maxim[1],'για k_neighbors=',nei[1],'\n')
    elif thes==2:
        print('scaled_Standard και kNN με cross validation έχω f_macro=',maxim[2],'για k_neighbors=',nei[2],'\n')
    elif thes==3:
        print('pca με n_components=',pc[3],' και kNN με cross validation έχω f_macro=',maxim[3],'για k_neighbors=',nei[3],'\n')
    elif thes==4:
        print('minmax και pca και kNN με n_components=',pc[4],' με cross validation έχω f_macro=',maxim[4],'για k_neighbors=',nei[4],'\n')
    elif thes==5:
        print('scaled_Standard και pca με n_components=',pc[5],' και kNN με cross validation έχω f_macro=',maxim[5],'για k_neighbors=',nei[5],'\n')
    elif thes==6:
        print('Vthreshold με value=',th[6],' και kNN με cross validation έχω f_macro=',maxim[6],'για k_neighbors=',nei[6],'\n')
    elif thes==7:
        print('Vthreshold με value=',th[7],' και min_max και kNN με cross validation έχω f_macro=',maxim[7],'για k_neighbors=',nei[7],'\n')
    elif thes==8:
        print('Vthreshold με value=',th[8],' και scaled_Standard και kNN με cross validation έχω f_macro=',maxim[8],'για k_neighbors=',nei[8],'\n')
    elif thes==9:
        print('Vthreshold με value=',th[9],' και min_max και pca με n_components=',pc[9],'  kNN με cross validation έχω f_macro=',maxim[9],'για k_neighbors=',nei[9],'\n')
    elif thes==10:
        print('Vthreshold με value=',th[10],' και scaled_Standard και pca με n_components=',pc[10],'  kNN με cross validation έχω f_macro=',maxim[10],'για k_neighbors=',nei[10],'\n')
    return


# In[61]:


def pipe_grid_micro(a,b,c):
    maxim=[0,0,0,0,0,0,0,0,0,0,0]
    nei=[0,0,0,0,0,0,0,0,0,0,0]
    pc=[0,0,0,0,0,0,0,0,0,0,0]
    th=[0,0,0,0,0,0,0,0,0,0,0]
    mx=0
    thes=0
    knn_micro(maxim,nei,pc,th)
    print('9%')
    minmax_knn_micro(maxim,nei,pc,th)
    print('18%')
    standard_knn_micro(maxim,nei,pc,th)
    print('27%')
    pca_knn_micro(maxim,nei,pc,th)
    print('36%')
    minmax_pca_knn_micro(maxim,nei,pc,th)
    print('45%')
    standard_pca_knn_micro(maxim,nei,pc,th)
    print('54%')
    thres_knn_micro(maxim,nei,pc,th,a,b,c)
    print('63%')
    thres_minmax_knn_micro(maxim,nei,pc,th,a,b,c)
    print('72%')
    thres_standard_knn_micro(maxim,nei,pc,th,a,b,c)
    print('81%')
    thres_minmax_pca_knn_micro(maxim,nei,pc,th,a,b,c)
    print('90%')
    thres_standard_pca_knn_micro(maxim,nei,pc,th,a,b,c)
    print('99%')
    print('\n')
    print('\n')
    print('\n')
    print("!!!!!!!!!")
    print("H βέλτιστη αρχιτεκτονική για f1_micro ειναι η εξής:\n")
    for i in range(11):
        if(maxim[i]>mx):
            thes=i
            mx=maxim[i]
    if thes==0:
        print('kNN με cross validation έχω f_micro=',maxim[0],'για k_neighbors=',nei[0],'\n')
    elif thes==1:
        print('min_max και kNN με cross validation έχω f_micro=',maxim[1],'για k_neighbors=',nei[1],'\n')
    elif thes==2:
        print('scaled_Standard και kNN με cross validation έχω f_micro=',maxim[2],'για k_neighbors=',nei[2],'\n')
    elif thes==3:
        print('pca με n_components=',pc[3],' και kNN με cross validation έχω f_micro=',maxim[3],'για k_neighbors=',nei[3],'\n')
    elif thes==4:
        print('minmax και pca και kNN με n_components=',pc[4],' με cross validation έχω f_micro=',maxim[4],'για k_neighbors=',nei[4],'\n')
    elif thes==5:
        print('scaled_Standard και pca με n_components=',pc[5],' και kNN με cross validation έχω f_micro=',maxim[5],'για k_neighbors=',nei[5],'\n')
    elif thes==6:
        print('Vthreshold με value=',th[6],' και kNN με cross validation έχω f_micro=',maxim[6],'για k_neighbors=',nei[6],'\n')
    elif thes==7:
        print('Vthreshold με value=',th[7],' και min_max και kNN με cross validation έχω f_micro=',maxim[7],'για k_neighbors=',nei[7],'\n')
    elif thes==8:
        print('Vthreshold με value=',th[8],' και scaled_Standard και kNN με cross validation έχω f_micro=',maxim[8],'για k_neighbors=',nei[8],'\n')
    elif thes==9:
        print('Vthreshold με value=',th[9],' και min_max και pca με n_components=',pc[9],'  kNN με cross validation έχω f_micro=',maxim[9],'για k_neighbors=',nei[9],'\n')
    elif thes==10:
        print('Vthreshold με value=',th[10],' και scaled_Standard και pca με n_components=',pc[10],'  kNN με cross validation έχω f_micro=',maxim[10],'για k_neighbors=',nei[10],'\n')
    return


# Πάμε να βρούμε τη διακύμανση του set μας έτσι ώστε να προσαρμόσουμε κατάλληλα το κάλεσμα της Variance Threshold

# In[62]:


train_variance = train.var(axis=0)
print(train_variance)
print(np.max(train_variance))


# In[63]:


start_time = time.time()


# In[64]:


pipe_grid_macro(1,1000,100)


# In[65]:


pipe_grid_micro(1,1000,100)


# In[66]:


pipe_grid_macro(0.1,1,0.05)


# In[67]:


pipe_grid_micro(0.1,1,0.05)


# In[68]:


pipe_grid_macro(1000,10000000,100000)


# In[69]:


pipe_grid_micro(1000,10000000,100000)


# In[71]:


pipe_grid_macro(0,0.1,0.01)


# In[72]:


pipe_grid_micro(0,0.1,0.01)


# In[73]:


print("Για την εύρεση του κατάλληλου fit για αυτο το training set χρειάστηκαν : %s seconds" % (time.time() - start_time))


# Όπως ήταν αναμενόμενο για μεγάλες τιμές του threshold δεν εφαρμόζεται αυτη η βελτιστοποίηση , καθώς χάνουμε μεγάλο ποσοστό απο χαρακτηριστικά που τα είχαμε ορίσει 0 ή 1 με την getdummies , αρα εχουν μικρη διακύμανση.
# 
# Επίσης βλέπουμε να επικρατεί πάντα το scaled Standard αντι του min max που είναι και αυτό αναμενόμενο , αφού ειναι απαραίτητη αυτη η βελτιστοποίηση για να δουλέψει καλύτερα ένας ταξινομητής.
# 
# Έπειτα για μια καλύτερη λειτουργία έχουμε το pca που βοηθάει πάρα πολύ για την μείωση της διαστατικότητας.

# Επιλέγουμε σαν καταλληλότερο fit την τελευταία επιλογή με Vthreshold με value= 0.08  και scaled_Standard και kNN με cross validation έχω  για k_neighbors= 19 και πάμε να την εφαρμόσουμε 

# In[118]:


fwei1=[0,0,0,0,0,0]
fmac1=[0,0,0,0,0,0]
fmic1=[0,0,0,0,0,0]
train_fin,test_fin=variance(train,test,0.08)
train_fin,test_fin=scaled_Standard(train_fin,test_fin)
num1=cross_macro(train_fin)


# In[119]:


def function1(train_in,train_labels_in,test_in,test_labels_in,num,xron,fwei1,fmac1,fmic1):
    
    credit_accuracy = {}

    dc_uniform = DummyClassifier(strategy="uniform")

# με τη μέθοδο fit "εκπαιδεύουμε" τον ταξινομητή στο σύνολο εκπαίδευσης (τα χαρακτηριστικά και τις ετικέτες τους)
    start_time = time.time()
    model = dc_uniform.fit(train_in, train_labels_in)
    xron[0]=(time.time() - start_time)
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
    
    fmac1[0]=f1_score(test_labels_in,preds,average='macro')
    fmic1[0]=f1_score(test_labels_in,preds,average='micro')
    fwei1[0]=f1_score(test_labels_in,preds,average='weighted')
    
    
    dc_constant_1 = DummyClassifier(strategy="constant", constant=1)
    start_time = time.time()
    model = dc_constant_1.fit(train_in, train_labels_in)
    xron[1]=(time.time() - start_time)
    preds = dc_constant_1.predict(test_in)
    credit_accuracy['constant 1'] = dc_constant_1.score(test_in, test_labels_in)


#################
    print ('Classification report for Dummy Classifier (constant-1)')
    cr_dummy_const1 = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_const1)

    scores_weighted['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Const1']=precision_recall_fscore_support(test_labels_in,preds,average='micro')


    print ('Confusion Matrix for Dummy Classifier (constant-1)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_const1 = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_const1))
    
    fmac1[1]=f1_score(test_labels_in,preds,average='macro')
    fmic1[1]=f1_score(test_labels_in,preds,average='micro')
    fwei1[1]=f1_score(test_labels_in,preds,average='weighted')
    
    
    dc_constant_0 = DummyClassifier(strategy="constant", constant=0)
    start_time = time.time()
    model = dc_constant_0.fit(train_in, train_labels_in)
    xron[2]=(time.time() - start_time)
    preds = dc_constant_0.predict(test_in)
    credit_accuracy['constant 0'] = dc_constant_0.score(test_in, test_labels_in)

#################
    print ('Classification report for Dummy Classifier (constant-0)')
    cr_dummy_const0 = classification_report(test_labels_in, preds,target_names = label_names)
    print (cr_dummy_const0)

    scores_weighted['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='weighted')
    scores_macro['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='macro')
    scores_micro['Dummy-Const0']=precision_recall_fscore_support(test_labels_in,preds,average='micro')

    print ('Confusion Matrix for Dummy Classifier (constant-0)')
    print (confusion_matrix(test_labels_in, preds))

    acc_dummy_const0 = 100*accuracy_score(test_labels_in,preds)
    print ('Accuracy percentage of this classifier is %.3f %%\n' % (acc_dummy_const0))
    
    fmac1[2]=f1_score(test_labels_in,preds,average='macro')
    fmic1[2]=f1_score(test_labels_in,preds,average='micro')
    fwei1[2]=f1_score(test_labels_in,preds,average='weighted')
    
    
    dc_most_frequent = DummyClassifier(strategy="most_frequent")
    start_time = time.time()
    model = dc_most_frequent.fit(train_in, train_labels_in)
    xron[3]=(time.time() - start_time)
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
    
    fmac1[3]=f1_score(test_labels_in,preds,average='macro')
    fmic1[3]=f1_score(test_labels_in,preds,average='micro')
    fwei1[3]=f1_score(test_labels_in,preds,average='weighted')
    
    
    
    dc_stratified = DummyClassifier(strategy="stratified")
    start_time = time.time()
    model = dc_stratified.fit(train_in, train_labels_in)
    xron[4]=(time.time() - start_time)
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
    
    fmac1[4]=f1_score(test_labels_in,preds,average='macro')
    fmic1[4]=f1_score(test_labels_in,preds,average='micro')
    fwei1[4]=f1_score(test_labels_in,preds,average='weighted')
    
    from sklearn.neighbors import KNeighborsClassifier
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=num)
    xron[5]=(time.time() - start_time)
    knn.fit(train_in, train_labels_in)
    knn_preds = knn.predict(test_in)


#################
    print ('Classification report for opt kNN')
    cr_knn_no = classification_report(test_labels_in,knn_preds, target_names=label_names)
    print (cr_knn_no)

    scores_weighted['kNN-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='weighted')
    scores_macro['kNN-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='macro')
    scores_micro['kNN-opt']=precision_recall_fscore_support(test_labels_in,knn_preds,average='micro')

    print ('Confusion Matrix for optimized kNN')
    print (confusion_matrix(test_labels_in, knn_preds))

    acc_knn_no = 100*accuracy_score(test_labels_in,knn_preds)
    print ('\nAccuracy percentage of this classifier is %.3f %%\n' % (acc_knn_no))
    
    fmac1[5]=f1_score(test_labels_in,knn_preds,average='macro')
    fmic1[5]=f1_score(test_labels_in,knn_preds,average='micro')
    fwei1[5]=f1_score(test_labels_in,knn_preds,average='weighted')
    
    
    
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


# In[120]:


xron=[0,0,0,0,0,0]
function1(train_fin,train_labels,test_fin,test_labels,num1,xron,fwei1,fmac1,fmic1)


# Αρχικά παρατηρούμε ότι ο Dummy Classifier δεν έχει βελτιωθεί καθόλου , που ηταν και αναμενόμενο
# Στη συνέχεια παρατηρούμε μια βελτίωση της τάξης του 20% στον kNN που είναι μια αισθητή μεταβολή

# In[96]:


print("Ο χρόνος για κάθε fit φαίνεται στον παρακάτω πίνακα ξεκινώντας απο τον:\n")
print('Dummy-Uniform            ',xron[0],'sec\n')
print('Dummy-Const1             ',xron[1],'sec\n')
print('Dummy-Const0             ',xron[2],'sec\n')
print('Dummy-Most_Freq          ',xron[3],'sec\n')
print('Dummy-Strat              ',xron[4],'sec\n')
print('kNN-opt                  ',xron[5],'sec\n')


# Αντίστοιχα με την αποπάνω σειρά εμφανίζεται η διαφορά στην απόδοση πριν και μετα την βελτιστοποιηση για τις μετρικες
# * f1-macro
# * f1_micro
# * f1_weighted

# In[122]:


for i in range(6):
    print(fmac1[i]-fmac[i],'\n')
    print(fmic1[i]-fmic[i],'\n')
    print(fwei1[i]-fwei[i],'\n')
    print('--------------------')


# Φαίνεται και εδώ ότι οι μεγάλες βελτιώσεις ειναι στη τελευταία περίπτωση , οι υπόλοιπες μικροαλλαγές στηρίζονται στη τυχαιότητα του dummy

# O χρόνος εκτέλεσης του kNN έχει διαφόρα καθώς λαμβάνεται υπόψη η υπερπαράμετρος n_neighbors 

# Για τις υπόλοιπες μετρικές σχολιάζουμε παραπάνω τα αναμενόμενα αποτελέσματα , όταν προκύπτει η τελική αρχιτεκτονική .
