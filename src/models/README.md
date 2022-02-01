Qui troviamo le architetture. 

Ogni architettura deve essere contenuta all'interno di una specifica directory denominata <nome_architettura> 
In particolare il tree è:

├───<nome_architettura>
	├───G1.py
	├───G2.py
	└───D.py

G1.py, G2.py, D.py, ereditano dal Model_Template. In particolare quest'ultima definisce i metodi
astratti da implementare nelle classi ereditarie.
Inoltre, ogni classe ereditante (G1.py, G2.py, D.py,), deve richiamere il costruttore della 
super classe poichè questultima richiama i metodi build_model e optimizer.
Per utilizzare l'architettura basta inserire il <nome_architettura> in src/CONFIG alla variabile ARCHITECTURE.
