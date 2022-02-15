Here it is possible to find the existing architectures in the framework or, if necessary, to define new ones.

Each architecture must be contained in a specific directory called with the properly name of architecture. 
In particular the tree is:

```
├───[architecture_name]
	├───G1.py
	├───G2.py
	└───D.py
```

G1.py, G2.py, D.py, extend the Model_Template.py. In particular, Model_Template defines the abstract methods to be implemented in the extended classes.
In addition, each extended class (G1.py, G2.py, D.py,), must call the super_class custructor. 
In fact, this method will invocke the <i>build_model</i> and <i>optimizer</i> methods.
To use the architecture in the framework, simply enter the [architecture_name] in src/CONFIG.py to the <b>ARCHITECTURE</b> variable.

