MLP Project - Cifar-10 Classification
Student: Dimitrios Netskas
-------------------------------------

Περιεχόμενα Φακέλου:

1. evaluate_model.py
   - Python script για γρήγορη αξιολόγηση (Inference only).
   - Φορτώνει τα έτοιμα βάρη και τυπώνει το Accuracy στο Test Set.
   - Δεν απαιτεί εκπαίδευση.

2. best_model.npz
   - Τα αποθηκευμένα βάρη του καλύτερου μοντέλου (Experiment 9, Run 11).
   - Χρησιμοποιείται από το evaluate_model.py.

3. mlp.ipynb (Jupyter Notebook)
   - Ο πλήρης κώδικας της εργασίας.
   - Περιέχει όλη τη διαδικασία: Data Loading, Model Definition, Training Loop, και Visualization.
   - Περιλαμβάνει τα logs και τα διαγράμματα (plots) από τα πειράματα.
   - Προσοχή: Αν εκτελεστεί, θα ξεκινήσει εκπαίδευση από την αρχή (χρονοβόρο).

4. requirements.txt
   - Λίστα εξαρτήσεων/βιβλιοθηκών.

-------------------------------------
Οδηγίες Εγκατάστασης & Εκτέλεσης (Windows):

Βήμα 1: Δημιουργία Virtual Environment (Προαιρετικά)
   > python -m venv venv
   > venv\Scripts\activate  (ή .\venv\Scripts\Activate στο PowerShell)

Βήμα 2: Εγκατάσταση Βιβλιοθηκών
   > pip install -r requirements.txt

Βήμα 3: Εκτέλεση Αξιολόγησης (Recommended)
   Για να δείτε το τελικό Accuracy (57.22%) χωρίς αναμονή:
   > python evaluate_model.py

Βήμα 4: Επισκόπηση Κώδικα Εκπαίδευσης
   Για να δείτε τον κώδικα εκπαίδευσης και τα γραφήματα, ανοίξτε το 
   'mlp.ipynb' με Jupyter Notebook, VS Code ή Google Colab.

