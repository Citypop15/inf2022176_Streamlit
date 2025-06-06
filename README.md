## Εφαρμογή Streamlit για ανάλυση δεδομένων βιομοριακής βιολογίας - Εργασία στο μάθημα Τεχνολογία Λογισμικού

Μια διαδραστική εφαρμογή Streamlit για ανάλυση δεδομένων και μηχανική μάθηση σε σύνολα δεδομένων βιομοριακής βιολογίας.  
Κατασκευασμένη με Python, pandas, scikit-learn, και φορητή με Docker.

---

## Χαρακτηριστικά

- Ανέβασμα και ανάλυση δεδομένων CSV
- Εκτέλεση clustering με KMeans
- Εκπαίδευση μοντέλων ταξινόμησης με Random Forest
- Πρόβλεψη σε νέα δεδομένα χρησιμοποιώντας trained model
- Dark theme UI με προσαρμοσμένο στυλ
- Φορητό: λειτουργεί μέσω Docker

---

## Εκτέλεση της Εφαρμογής με Docker

1. Βεβαιωθείτε ότι το Docker Desktop είναι εγκατεστημένο και λειτουργεί  
2. Κάνετε clone ή κατεβάστε αυτό το repo, και μετά ανοίξτε ένα τερματικό στον φάκελο του project
3. Τρέξτε τις παρακάτω εντολές bash στο terminal
4. Μετά, επισκεφτείτε http://localhost:8501 στο browser σας
   
```bash
docker build -t molecular-bio-app .
docker run --rm -p 8501:8501 molecular-bio-app


