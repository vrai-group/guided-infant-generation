Here it is possible to define new architectures.
Each architecture must be contained in a specific directory called with the [architecture name].
<br>
Each directory must be structered ad follows:

```
├───[architecture name]
	├───G1.py
	├───G2.py
	└───D.py
```

G1.py, G2.py, D.py, extend the Model_Template.py. In particular, Model_Template defines the abstract methods to be implemented in the extended classes.
In addition, each extended class (G1.py, G2.py, D.py,), must call the super_class custructor. 
In fact, this method will invocke the <i>build_model</i> and <i>optimizer</i> methods.
To use the architecture in the framework, simply enter the [architecture name] in the <a href="https://github.com/GiuseppeCannata/BabyPoseGuided/blob/dbd146c1ce4e8303ba2110b471efb4ccf83cbae9/src/CONFIG.py#L20">ARCHITECTURE</a> variable.
<br><br>
As examples you can consider the <a href="./bibranch">./bibranch</a>, <a href="./mono">./mono</a> architectures.

