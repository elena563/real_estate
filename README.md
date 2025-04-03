# App Web per la previsione dei prezzi immobiliari

### Panoramica
Questo progetto è un'applicazione Web creata utilizzando Streamlit che prevede il prezzo al metro quadro degli immobili nel distretto di Sindian di Nuova Taipei City, Taiwan. Gli utenti possono scegliere come ottenere la previsione tramite due modelli di regressione lineare con diverse covariate:
- **latitudine** e **longitudine** per il primo modello
- **età** della casa, **vicinanza alla stazione MRT più vicina** e **numero di minimarket** nella zona per il secondo.

### Tecnologie utilizzate
- Streamlit: per l'interfaccia utente.
- Scikit-learn: per la creazione del modello di machine learning (regressione lineare).
- Pandas: per la manipolazione dei dati.
- Pickle: per il salvataggio e il caricamento dei modelli.

### Installazione
**1. Clona il repository**
```git clone https://github.com/elena563/real_estate```

**2. Crea un ambiente virtuale e installa le dipendenze**

**3. Esegui lo script e apri l'interfaccia dell'app**
```python train_model.py```
```streamlit run UI.py```